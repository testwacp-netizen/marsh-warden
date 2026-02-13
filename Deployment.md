# Streamlit Cloud Deployment Guide

This guide provides step-by-step instructions for deploying the CircularIQ application to Streamlit Cloud. Following these steps will help you resolve common deployment issues.

## ❗️ Most Common Error: `KeyError`

If your deployed application shows a `KeyError` (e.g., `KeyError: 'st.secrets has no key "hf_token"'`), it means the secrets in your Streamlit Cloud dashboard are missing or incorrect. This is a **deployment configuration issue**, not a bug in the code. The solution is always to fix the secrets in the Streamlit Cloud settings.

The app now includes a startup check and will show a user-friendly error message listing exactly which secrets are missing.

## 1. Pre-deployment Checklist

Before you begin, ensure you have:

- A GitHub account with the application code pushed to a repository.
- A Streamlit Cloud account.
- Your Google OAuth credentials (`client_id` and `client_secret`).
- Your Hugging Face API Token (`hf_token`).

## 2. Required Data File

The application relies on the pre-built index file to function:

-   **`pdf_index_enhanced.pkl`**: This file contains the knowledge base for the chatbot. You must ensure this file is tracked by Git and is present in your GitHub repository. **The `pdf_files` directory is not needed for deployment.**

## 3. Google OAuth Configuration

Your application uses Google for authentication. You must configure the OAuth consent screen in the Google Cloud Console.

1.  Go to the **[Google Cloud Console](https://console.cloud.google.com/)**.
2.  Navigate to **APIs & Services -> Credentials**.
3.  Find your OAuth 2.0 Client ID.
4.  Under **Authorized redirect URIs**, you **must** add the URL of your deployed Streamlit app.
    -   The URL will be available after your first deployment (e.g., `https://your-app-name.streamlit.app/`).
    -   You may need to deploy once, get the URL, and then add it here for the login to work correctly.

## 4. How to Set Up Secrets Correctly

Follow these steps exactly to configure your secrets in Streamlit Cloud:

1.  In this repository, open the **`secrets.toml`** file. This file is a template for all the secrets your application needs.

2.  In a separate browser tab, go to your app's page on Streamlit Cloud and click **"Manage app"**.

3.  Navigate to the **"Secrets"** management section.

4.  **Copy the entire contents** of the `secrets.toml` file and **paste it into the secrets editor** on Streamlit Cloud.

5.  **Replace the placeholder values** (e.g., `"YOUR_HF_TOKEN_HERE"`) with your **actual secret values**.

Your final secrets in the Streamlit Cloud editor should look like this, but with your real values:
```toml
# This is an example. Use your actual secrets.

# Hugging Face API Token
hf_token = "hf_..."

# Google OAuth Credentials
client_id = "....apps.googleusercontent.com"
client_secret = "GOCSPX-..."

# The redirect URI for your Streamlit app.
# This MUST match the URI configured in your Google Cloud Console.
redirect_uri = "https://your-app-name.streamlit.app/"
```
6. Click **"Save"**. After saving, it is highly recommended to **reboot your app** from the dashboard to ensure the new secrets are loaded.

## 5. Deployment Steps

1.  **Log in to Streamlit Cloud.**
2.  Click **"New app"** and select your repository, branch, and the main file (`streamlit_app.py`).
3.  Click **"Deploy!"**.
4.  Once the app deploys (it will likely show an error initially), go to the app's settings and **add the secrets** as described in Section 4.
5.  Ensure your **Google Cloud Console has the correct redirect URI**.
6.  **Reboot the app** from the Streamlit Cloud dashboard.

By following these updated steps, your application should deploy successfully.
