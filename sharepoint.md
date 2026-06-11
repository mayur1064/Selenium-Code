# SharePoint Upload Utility

A Python command-line utility to upload HTML pages and attachments to SharePoint Online. HTML pages are uploaded with an `.aspx` extension so SharePoint renders them natively (not as downloadable files).

## Features

- **Upload HTML as rendered SharePoint pages** — converts `.html` to `.aspx` and places them in the `SitePages` library
- **Upload attachments** — upload any files (images, CSS, JS, PDFs, etc.) to any document library
- **Flexible authentication** — supports both interactive **Device Code Flow** and automated **Client Credentials Flow**
- **Microsoft Graph API** — uses the modern Graph API, no legacy SOAP/web services needed

---

## Prerequisites

1. **Python 3.8+** installed
2. **Azure AD App Registration** with the following:
   - API Permissions (delegated or application):
     - `Sites.ReadWrite.All`
     - `Files.ReadWrite.All`
   - If using Device Code Flow: set **Mobile and desktop applications** redirect URI to `https://login.microsoftonline.com/common/oauth2/nativeclient`
   - If using Client Credentials: generate a **client secret**

> **Tip**: You can register the app at [Azure Portal → App Registrations](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade).

---

## Installation

```bash
# Clone or copy the files, then:
pip install -r requirements.txt
```

---

## Configuration

1. Copy the example config:
   ```bash
   cp config.example.json config.json
   ```

2. Edit `config.json` with your values:

```json
{
    "auth": {
        "client_id": "11111111-2222-3333-4444-555555555555",
        "tenant_id": "your-tenant.onmicrosoft.com",
        "client_secret": "",
        "use_device_code": true
    },
    "sharepoint": {
        "site_url": "https://your-tenant.sharepoint.com/sites/YourSite",
        "site_name": "YourSite",
        "server_relative_url": "/sites/YourSite"
    },
    "pages": {
        "library": "SitePages",
        "content_type": "aspx"
    },
    "attachments": {
        "library": "Shared Documents"
    }
}
```

| Field | Description |
|---|---|
| `auth.client_id` | Your Azure AD app's Application (client) ID |
| `auth.tenant_id` | Your tenant ID or domain (e.g. `contoso.onmicrosoft.com`) |
| `auth.client_secret` | Leave empty for device code; required for app-only auth |
| `auth.use_device_code` | `true` for interactive browser login; `false` for automated auth |
| `sharepoint.site_url` | Full URL of your SharePoint site |
| `pages.library` | Library where `.aspx` pages are stored (default: `SitePages`) |
| `attachments.library` | Default library for attachment uploads (default: `Shared Documents`) |

---

## Usage

### Upload an HTML Page

```bash
python sharepoint_uploader.py --page path/to/index.html --page-title "Home Page"
```

This uploads the HTML as `index.aspx` to the **SitePages** library. SharePoint will render it as a page.

### Upload Attachments

```bash
python sharepoint_uploader.py --attachments logo.png report.pdf data.csv
```

Uploads files to the default attachments library (from config) at the root folder.

### Upload Attachments to a Specific Folder

```bash
python sharepoint_uploader.py --attachments img/banner.jpg doc/summary.pdf \
    --attach-folder "Marketing/Q2-2026"
```

### Upload Attachments to a Custom Library

```bash
python sharepoint_uploader.py --attachments style.css app.js \
    --attach-library "SiteAssets"
```

### Do Both — Page + Attachments

```bash
python sharepoint_uploader.py \
    --page landing.html \
    --page-title "Landing Page" \
    --attachments css/main.css js/app.js images/hero.jpg \
    --attach-library "SiteAssets"
```

### Custom Config File

```bash
python sharepoint_uploader.py --config prod-config.json --page index.html
```

### All Options

```
usage: sharepoint_uploader.py [-h] [-c CONFIG] [--page PAGE]
                              [--page-title PAGE_TITLE] [--page-name PAGE_NAME]
                              [--attachments ATTACHMENTS [ATTACHMENTS ...]]
                              [--attach-folder ATTACH_FOLDER]
                              [--attach-library ATTACH_LIBRARY] [-v]

Options:
  -c, --config          Path to JSON config file (default: config.json)
  --page                HTML file to upload as a rendered SharePoint page
  --page-title          Display title for the page (default: filename stem)
  --page-name           Override the .aspx filename (without extension)
  --attachments         One or more file paths to upload as attachments
  --attach-folder       Target folder in the attachments library
  --attach-library      Library name override for attachments
  -v, --verbose         Enable debug logging
```

---

## Authentication Modes

### 1. Device Code Flow (Interactive — Recommended)

Set `"use_device_code": true` in your config (no client secret needed).

When you run the script, you'll see:

```
============================================================
  AUTHENTICATION REQUIRED
============================================================

  Go to:  https://microsoft.com/devicelogin
  Enter:  ABC123DEF

  Waiting for you to sign in...
```

Open the URL in a browser, enter the code, and sign in. The script will proceed automatically.

### 2. Client Credentials Flow (Automated — for scripts/CI)

Set `"use_device_code": false` and provide a `client_secret`. The app authenticates silently — ideal for automation, scheduled tasks, or CI/CD pipelines.

---

## How It Works

1. **Authentication**: The script acquires an OAuth token via MSAL.
2. **Site resolution**: It resolves the SharePoint site ID from the URL.
3. **Drive lookup**: Finds the correct document library (drive).
4. **Page upload**: HTML files are wrapped in a minimal SharePoint page template (`<%@ Page %>`, `<asp:Content>`) and uploaded as `.aspx` to the SitePages library.
5. **Attachment upload**: Arbitrary files are uploaded via `PUT /sites/{id}/drives/{drive}/root:/path:/content`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `config.json not found` | Copy `config.example.json` → `config.json` |
| `Authentication failed` | Verify your `client_id` and `tenant_id`. Check app permissions in Azure Portal. |
| `Library 'X' not found` | The library name must match exactly (case-insensitive). Check available libraries in your SharePoint site. |
| `403 Forbidden` | Ensure the Azure AD app has `Sites.ReadWrite.All` and `Files.ReadWrite.All` granted + admin consented. |
| `Page doesn't render` | Verify the file was uploaded as `.aspx` to the SitePages library. Modern SharePoint may require you to **check in** or **publish** the page after upload. |
| HTML content not showing | The script wraps your HTML in a SharePoint-compatible page template. If you have complex HTML, ensure it's valid and doesn't conflict with SharePoint's master page. |

### Granting Admin Consent

If using application permissions (client credentials), an admin must grant consent:

1. Go to **Azure Portal → App Registrations → Your App → API Permissions**
2. Click **Grant admin consent for {tenant}**
3. Confirm

---

## Limitations

- **Modern SharePoint pages**: This utility creates **classic wiki/ASPX pages**. For modern SharePoint pages with web parts, use the SharePoint PnP PowerShell or the SharePoint Framework (SPFx).
- **HTML complexity**: Very complex HTML with external scripts/stylesheets may need the referenced assets uploaded to **SiteAssets** and paths adjusted.
- **Page approval**: Some SharePoint sites require content approval — pages may appear as **draft** until published.

---

## License

MIT
