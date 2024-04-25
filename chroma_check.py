import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
  settings=Settings(chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",chroma_client_auth_credentials="admin:admin"))
print(client.heartbeat())  # this should work with or without authentication - it is a public endpoint

print(client.get_version())  # this should work with or without authentication - it is a public endpoint

print(client.list_collections())  # this is a protected endpoint and requires authentication
