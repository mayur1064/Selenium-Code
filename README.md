curl -k -s -w "\nHTTP: %{http_code}\n" -H "Authorization: Bearer YOUR_PAT" "https://central-confluence.company.net/rest/api/content/PAGE_ID"
