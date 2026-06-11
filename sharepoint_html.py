#!/usr/bin/env python3
"""
SharePoint Upload Utility
=========================
Uploads HTML pages (as .aspx for rendering) and attachments to SharePoint Online.

Supports:
  - Device Code Flow (interactive, no client secret needed)
  - Client Credentials Flow (app-only, automated)

Authentication:
  Requires an Azure AD app registration with appropriate permissions.
  See README.md for setup instructions.
"""

import argparse
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import msal

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sp-uploader")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SCOPES = ["https://graph.microsoft.com/.default",
          "Sites.ReadWrite.All",
          "Files.ReadWrite.All",
          "Sites.Manage.All"]


# ===================================================================
# Authentication
# ===================================================================
class SharePointAuth:
    """Handles authentication with Azure AD using MSAL."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.client_id = config["auth"]["client_id"]
        self.tenant_id = config["auth"]["tenant_id"]
        self.client_secret = config["auth"].get("client_secret", "")
        self.use_device_code = config["auth"].get("use_device_code", True)
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"

    def acquire_token(self) -> str:
        """Acquire an access token, preferring device-code flow for interactive use."""
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=self.authority,
            client_credential=self.client_secret or None,
        )

        if self.use_device_code or not self.client_secret:
            return self._device_code_flow(app)
        return self._client_credentials_flow(app)

    def _device_code_flow(self, app: msal.ConfidentialClientApplication) -> str:
        """Interactive device-code authentication."""
        # Device code flow works best with PublicClientApplication,
        # but ConfidentialClientApplication also supports it.
        public_app = msal.PublicClientApplication(
            self.client_id, authority=self.authority
        )
        flow = public_app.initiate_device_flow(scopes=SCOPES)
        if not flow:
            raise RuntimeError("Failed to initiate device code flow.")

        print("\n" + "=" * 60)
        print("  AUTHENTICATION REQUIRED")
        print("=" * 60)
        print(f"\n  Go to:  {flow['verification_uri']}")
        print(f"  Enter:  {flow['user_code']}")
        print("\n  Waiting for you to sign in...\n")

        result = public_app.acquire_token_by_device_flow(flow)
        if "access_token" in result:
            log.info("Authentication successful.")
            return result["access_token"]

        error = result.get("error_description", result.get("error", "Unknown error"))
        raise RuntimeError(f"Authentication failed: {error}")

    def _client_credentials_flow(self, app: msal.ConfidentialClientApplication) -> str:
        """Non-interactive client-credentials (app-only) authentication."""
        result = app.acquire_token_for_client(scopes=SCOPES)
        if "access_token" in result:
            log.info("Authenticated via client credentials.")
            return result["access_token"]
        error = result.get("error_description", result.get("error", "Unknown error"))
        raise RuntimeError(f"Client credentials auth failed: {error}")


# ===================================================================
# SharePoint Client
# ===================================================================
class SharePointClient:
    """Handles SharePoint operations via Microsoft Graph API."""

    def __init__(self, config: Dict[str, Any], access_token: str) -> None:
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        self._site_id: Optional[str] = None
        self._drive_id: Optional[str] = None
        # Store the target library for pages and attachments
        self.pages_library = config["pages"].get("library", "SitePages")
        self.attachments_library = config["attachments"].get("library", "Shared Documents")

    # -- Site / Drive helpers -----------------------------------------------
    @property
    def site_id(self) -> str:
        if self._site_id is None:
            self._site_id = self._resolve_site_id()
        return self._site_id

    @property
    def drive_id(self) -> str:
        if self._drive_id is None:
            self._drive_id = self._resolve_drive_id()
        return self._drive_id

    def _resolve_site_id(self) -> str:
        """Resolve the SharePoint site ID from the site URL."""
        site_url = self.config["sharepoint"]["site_url"]
        hostname = site_url.split("/")[2]  # e.g. tenant.sharepoint.com
        site_path = site_url.split(hostname, 1)[1]  # e.g. /sites/MySite

        # Graph endpoint: /sites/{hostname}:/{site-path}
        url = f"{GRAPH_BASE}/sites/{hostname}:{site_path}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        site_id = resp.json()["id"]
        log.info("Resolved site ID: %s", site_id)
        return site_id

    def _resolve_drive_id(self) -> str:
        """Get the default document-library drive ID for the site."""
        url = f"{GRAPH_BASE}/sites/{self.site_id}/drives"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        drives = resp.json()["value"]
        if not drives:
            raise RuntimeError("No drives (document libraries) found on site.")
        drive_id = drives[0]["id"]
        log.info("Using drive: %s (ID: %s)", drives[0].get("name", "default"), drive_id)
        return drive_id

    # -- Resolve a drive by library name ------------------------------------
    def _get_drive_by_name(self, library_name: str) -> str:
        """Find a drive (document library) by its display name."""
        url = f"{GRAPH_BASE}/sites/{self.site_id}/drives"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        for drive in resp.json()["value"]:
            if drive.get("name", "").lower() == library_name.lower():
                return drive["id"]
        raise RuntimeError(
            f"Library '{library_name}' not found. Available: "
            + ", ".join(d.get("name", "?") for d in resp.json()["value"])
        )

    # -- Upload helpers -----------------------------------------------------
    def upload_file(
        self,
        local_path: str,
        target_folder: str,
        target_filename: Optional[str] = None,
        library_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a single file to a SharePoint document library.

        Args:
            local_path: Path to the local file.
            target_folder: Folder path within the library (e.g. '' for root).
            target_filename: Override the uploaded filename. Defaults to basename.
            library_name: Target library name. Uses default drive if omitted.

        Returns:
            API response as dict.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        filename = target_filename or local_path.name
        library = library_name or self.attachments_library
        drive = self._get_drive_by_name(library)

        # Build upload URL
        item_path = f"{target_folder}/{filename}".strip("/")
        upload_url = (
            f"{GRAPH_BASE}/sites/{self.site_id}/drives/{drive}"
            f"/root:/{item_path}:/content"
        )

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(local_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        log.info("Uploading: %s  ->  %s/%s", local_path.name, library, item_path)

        with open(local_path, "rb") as fh:
            headers = {**self.headers, "Content-Type": mime_type}
            resp = requests.put(upload_url, headers=headers, data=fh)

        if resp.status_code in (200, 201):
            data = resp.json()
            web_url = data.get("webUrl", data.get("@microsoft.graph.downloadUrl", ""))
            log.info("  ✓  Uploaded successfully: %s", web_url)
            return data
        else:
            log.error("  ✗  Upload failed [%s]: %s", resp.status_code, resp.text)
            resp.raise_for_status()

    def upload_as_page(
        self,
        html_path: str,
        page_title: Optional[str] = None,
        target_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload an HTML file as a SharePoint wiki/ASPX page in SitePages.

        The file is uploaded with an .aspx extension so SharePoint renders
        it as a page rather than treating it as a downloadable file.

        Args:
            html_path: Path to the local .html file.
            page_title: Title for the SharePoint page (metadata).
            target_filename: Override the .aspx filename (without extension).

        Returns:
            API response as dict.
        """
        local_path = Path(html_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        if local_path.suffix.lower() not in (".html", ".htm", ".aspx"):
            log.warning("File does not have .html/.aspx extension; uploading anyway.")

        base_name = target_filename or local_path.stem
        aspx_name = f"{base_name}.aspx"
        title = page_title or base_name

        log.info("Uploading HTML as page: %s -> %s", local_path.name, aspx_name)

        # Upload to SitePages library
        library = self.pages_library
        drive = self._get_drive_by_name(library)

        upload_url = (
            f"{GRAPH_BASE}/sites/{self.site_id}/drives/{drive}"
            f"/root:/{aspx_name}:/content"
        )

        # Read HTML content and wrap in SharePoint page markup if needed
        with open(local_path, "r", encoding="utf-8") as fh:
            html_content = fh.read()

        # Wrap bare HTML in a minimal SharePoint-compatible page if it isn't
        # already wrapped in <asp:Content> or similar controls.
        if "<asp:Content" not in html_content and "<%@ Page" not in html_content:
            html_content = _wrap_html_for_sharepoint(html_content, title)

        headers = {**self.headers, "Content-Type": "text/html; charset=utf-8"}
        resp = requests.put(upload_url, headers=headers, data=html_content.encode("utf-8"))

        if resp.status_code in (200, 201):
            data = resp.json()
            web_url = data.get("webUrl", "")
            log.info("  ✓  Page uploaded: %s", web_url)
            return data
        else:
            log.error("  ✗  Page upload failed [%s]: %s", resp.status_code, resp.text)
            resp.raise_for_status()

    def upload_attachments(
        self,
        attachments: List[str],
        target_folder: str = "",
        library_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Upload multiple attachment files to a SharePoint library.

        Args:
            attachments: List of local file paths.
            target_folder: Destination folder within the library.
            library_name: Target library. Uses attachments_library from config.

        Returns:
            List of API response dicts.
        """
        results = []
        for filepath in attachments:
            try:
                result = self.upload_file(
                    filepath,
                    target_folder=target_folder,
                    library_name=library_name,
                )
                results.append(result)
            except Exception as exc:
                log.error("Failed to upload %s: %s", filepath, exc)
        return results


# ===================================================================
# HTML wrapping helper
# ===================================================================
def _wrap_html_for_sharepoint(html_body: str, title: str) -> str:
    """Wrap raw HTML content in a minimal SharePoint wiki-page structure."""
    # Escape any occurrences of CDATA end markers
    safe_body = html_body.replace("]]>", "]]]]><![CDATA[>")

    return f"""<%@ Page Language="C#" %>
<%@ Register TagPrefix="WebPartPages"
    Namespace="Microsoft.SharePoint.WebPartPages"
    Assembly="Microsoft.SharePoint, Version=16.0.0.0, Culture=neutral,
    PublicKeyToken=71e9bce111e9429c" %>
<html dir="ltr">
<head>
    <meta name="WebPartPageExpansion" content="full" />
    <meta name="ProgId" content="SharePoint.WebPartPage.Document" />
    <title>{title}</title>
</head>
<body>
    <div id="contentBox">
        <![CDATA[
{safe_body}
        ]]>
    </div>
</body>
</html>
"""


# ===================================================================
# CLI
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload HTML pages and attachments to SharePoint Online.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a single HTML page as a rendered SharePoint page
  python sharepoint_uploader.py --page index.html --page-title "Home Page"

  # Upload attachments to a specific folder
  python sharepoint_uploader.py --attachments img/logo.png doc/report.pdf \\
      --attach-folder "Marketing/2026"

  # Do both at once
  python sharepoint_uploader.py --page landing.html \\
      --attachments css/style.css js/app.js images/bg.jpg \\
      --attach-library "SiteAssets"

  # Use a custom config file
  python sharepoint_uploader.py --config prod-config.json --page index.html
        """,
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to JSON configuration file (default: config.json).",
    )
    parser.add_argument(
        "--page",
        help="Path to an HTML file to upload as a rendered SharePoint page.",
    )
    parser.add_argument(
        "--page-title",
        help="Display title for the uploaded page (defaults to filename stem).",
    )
    parser.add_argument(
        "--page-name",
        help="Override the .aspx filename (without .aspx extension).",
    )
    parser.add_argument(
        "--attachments",
        nargs="+",
        help="One or more attachment file paths to upload.",
    )
    parser.add_argument(
        "--attach-folder",
        default="",
        help="Target folder in the attachments library (default: root).",
    )
    parser.add_argument(
        "--attach-library",
        help="Override the library for attachments (default from config).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        log.error("Configuration file not found: %s", config_path)
        log.error("Copy config.example.json to config.json and fill in your details.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Load config & authenticate
    # ------------------------------------------------------------------
    config = load_config(args.config)
    auth = SharePointAuth(config)
    token = auth.acquire_token()

    client = SharePointClient(config, token)

    # ------------------------------------------------------------------
    # Upload page
    # ------------------------------------------------------------------
    if args.page:
        log.info("=== Uploading HTML Page ===")
        try:
            client.upload_as_page(
                html_path=args.page,
                page_title=args.page_title,
                target_filename=args.page_name,
            )
        except Exception as exc:
            log.error("Page upload failed: %s", exc)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Upload attachments
    # ------------------------------------------------------------------
    if args.attachments:
        log.info("=== Uploading Attachments ===")
        results = client.upload_attachments(
            attachments=args.attachments,
            target_folder=args.attach_folder,
            library_name=args.attach_library,
        )
        log.info("Successfully uploaded %d/%d attachments.",
                 len(results), len(args.attachments))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if not args.page and not args.attachments:
        parser.print_help()
        print("\nNo files specified. Use --page and/or --attachments.")

    log.info("Done.")


if __name__ == "__main__":
    main()
