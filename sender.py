import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email and app password
sender_email = "aarishshah1@gmail.com"
receiver_email = "aarishshahmohsin50@gmail.com"  # Change this to the recipient's email
app_password = "nylq pjsu kxic zmbk"  

# Path to the file you want to send as content
file_path = "/tmp/aarish_job.log"  # Replace with your actual file path

# Read the content of the file
try:
    with open(file_path, "r") as file:
        file_content = file.read()  # Read the entire content of the file
except Exception as e:
    print(f"Error reading file: {e}")

# Message content
subject = "Job Status"
body = f"Job finished successfully.\n\nFile Content:\n\n{file_content}"

# Create the MIME message object
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject

# Add the body of the email
message.attach(MIMEText(body, "plain"))

# Gmail SMTP server details
smtp_server = "smtp.gmail.com"
smtp_port = 587  # TLS port

# Establish a secure SSL connection
context = ssl.create_default_context()

# Send the email using smtplib
try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls(context=context)  # Secure the connection
        server.login(sender_email, app_password)  # Login with your email and app password
        server.sendmail(sender_email, receiver_email, message.as_string())  # Send email with content from the file
    print("Email sent successfully!")
except Exception as e:
    print(f"Error sending email: {e}")
