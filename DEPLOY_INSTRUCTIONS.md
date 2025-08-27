# Deployment Instructions for DigitalOcean VPS

This guide will walk you through deploying the trash classifier web application on a DigitalOcean VPS (or any server running a modern Linux distribution like Ubuntu).

## Prerequisites

1.  A running VPS with root or `sudo` access.
2.  Your domain name pointing to the VPS's IP address (optional, you can use the IP address directly).
3.  Git, Python 3, and `python3-venv` installed on your server.
    ```bash
    sudo apt-get update
    sudo apt-get install git python3 python3-venv -y
    ```

## Step 1: Clone the Repository

Clone your repository to your home directory (or any other location you prefer).

```bash
git clone <your-repository-url>
cd <your-repo-directory-name>
```

## Step 2: Set up the Python Environment

Create a virtual environment and install the required packages.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure and Install the `systemd` Service

1.  **Edit the service file:**
    Open the `deploy/trash-classifier.service` file. You need to replace the placeholders:
    -   `User=your_user`: Change `your_user` to your username on the VPS.
    -   `WorkingDirectory=/path/to/your/repo`: Change this to the full path where you cloned the repository (e.g., `/home/your_user/trash-classifier-app`).
    -   `ExecStart=/path/to/your/repo/...`: Change the path here as well.

2.  **Copy the file to the systemd directory:**
    ```bash
    sudo cp deploy/trash-classifier.service /etc/systemd/system/
    ```

3.  **Start and enable the service:**
    ```bash
    # Reload systemd to recognize the new service
    sudo systemctl daemon-reload
    # Start the service
    sudo systemctl start trash-classifier.service
    # Enable the service to start on boot
    sudo systemctl enable trash-classifier.service
    ```

4.  **Check the status:**
    You can check if the service is running correctly with:
    ```bash
    sudo systemctl status trash-classifier.service
    ```
    You should also see a file named `app.sock` created in your project directory.

## Step 4: Configure and Install Nginx

1.  **Install Nginx:**
    ```bash
    sudo apt-get install nginx -y
    ```

2.  **Edit the Nginx configuration file:**
    Open `deploy/nginx.conf` and replace the placeholders:
    -   `server_name your_domain;`: Change `your_domain` to your actual domain or your server's IP address.
    -   `/path/to/your/repo`: Replace all instances of this with the full path to your project directory.

3.  **Copy the configuration to Nginx:**
    ```bash
    sudo cp deploy/nginx.conf /etc/nginx/sites-available/trash-classifier
    ```

4.  **Enable the site by creating a symbolic link:**
    ```bash
    sudo ln -s /etc/nginx/sites-available/trash-classifier /etc/nginx/sites-enabled
    # It's a good practice to remove the default config
    sudo rm /etc/nginx/sites-enabled/default
    ```

5.  **Test and restart Nginx:**
    ```bash
    # Test for syntax errors
    sudo nginx -t
    # If the test is successful, restart Nginx
    sudo systemctl restart nginx
    ```

## Step 5: Access Your Site

You should now be able to access your web application by navigating to `http://your_domain` (or your server's IP address) in your web browser.

---
### Notes on Security

-   For a production site, you should set up a firewall (`ufw`) and configure Nginx to use HTTPS with a free SSL certificate from Let's Encrypt. This configuration only covers HTTP on port 80.
