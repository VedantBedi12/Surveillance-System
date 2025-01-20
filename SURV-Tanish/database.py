from pymongo import MongoClient
import certifi

# Connection string
connection_string = "mongodb+srv://tanishjain1124:charu322@cluster0.jlwmj.mongodb.net/myDatabase?retryWrites=true&w=majority"

# Create MongoClient with SSL/TLS configuration
client = MongoClient(
    connection_string,
    tls=True,  # Enable TLS
    tlsCAFile=certifi.where()  # Use certifi's CA bundle for SSL certificates
)

# Test the connection
db = client.surveillance  # Replace "myDatabase" with your database name
collection = db.embeddings  # Replace "myCollection" with your collection name
