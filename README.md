curl -s -w "\nHTTP Status: %{http_code}\n" -u "your-email@company.com:YOUR_PAT" "https://confluence.your-company.com/rest/api/space"

curl -s -v -H "Authorization: Bearer YOUR_PAT" "https://central-confluence.company.net/rest/api/space" 2>&1 | grep -i "location:"
