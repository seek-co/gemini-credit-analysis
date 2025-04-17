# Local Setup Instruction

# Platform and Services Setup

## Github

Create github account, give me your account name & email, so that I can allow you access to the repo

  

If your Mac cannot use git, might need to install git

\- for macos: [https://git-scm.com/downloads/mac](https://git-scm.com/downloads/mac)

  

\*\* You can only do the following after being invited to access the github repo

At your local device terminal, go to a folder you decide to locate the project github repo.

  

And enter command

```plain
git clone https://github.com/kronecker-ai/credit-analysis.git
cd credit-analysis
```

  

Create a virtual environment for the project using this command (Presumably your MacOS has python 3.10)

\*\* Or you can use your coding IDE like VScode or Pycharm to do this if you're used to this step.

```plain
python3 -m venv path/to/your/virtual/env
```

replace the path with your actual path for virtual environment

  

and activate it

```plain
source venv-path/.../bin/activate
```

where `venv-path/.../` is the actual path of your virtual env folder that you just created. Just append `bin/activate` to the path.

  

Run command

```plain
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

  

Rename the `.env.example` file to `.env` within the github repo folder

  

## Google Cloud Platform

Go to `cloud.google.com` and create account

  

Create new project - click top left of your welcome page, possibly saying something like 'My First Project' and create a new project.

  

Make sure the created new project is selected

  

Follow the steps in this video to create a service account and get the JSON file credential

[https://www.youtube.com/watch?v=tamT\_iGoZDQ&ab\_channel=ShanmugamUdhaya](https://www.youtube.com/watch?v=tamT_iGoZDQ&ab_channel=ShanmugamUdhaya)

  

Once you get the service account JSON file, put all the credentials into the `.env` file, according to the names of the keys in the `.env.example` provided.

  

\*\* If encounter service account key creation is disabled error:

Follow this instruction here to enable:

[https://www.cubebackup.com/docs/tutorials/gcp-allow-service-account-key-creation/#:~:text=Assign%20Organization%20Policy%20Administrator%20role,-To%20set%20an&text=Sign%20in%20to%20the%20Google,Click%20the%20%2B%20GRANT%20ACCESS%20button](https://www.cubebackup.com/docs/tutorials/gcp-allow-service-account-key-creation/#:~:text=Assign%20Organization%20Policy%20Administrator%20role,-To%20set%20an&text=Sign%20in%20to%20the%20Google,Click%20the%20%2B%20GRANT%20ACCESS%20button).

  

You might have to go to "Organisation Policies". Sort the column "ID", scroll to find the IDs: `iam.disableServiceAccountCreation` and `iam.disableServiceAccountKeyCreation`. If they are disabled, choose to edit policy, and select `Override parent's policy`, and `Enforcement Off` .

  

\*\* Go to Enabled APIs & services, and enable all these in the list, if they are not enabled.

![](https://t9018625102.p.clickup-attachments.com/t9018625102/7bd1710d-c72f-4fc9-92fb-a9dc90d4df2e/Screenshot%20from%202025-04-10%2010-17-03.png)

  

  

Back to the top left hamburger menu, go to cloud storage, select buckets.

  

Create a new bucket, name the new bucket as: `ai-credit-analysis-bucket` , and tick 'enable hierarchical namespace on this bucket'.

for 'choose where to store your data', choose 'Region', and 'europe-west1 Belgium. The rest leave as it is.

  

Go back to the main page of your bucket, copy the exact bucket name, and put it in .env under the bucket name key.

  

Now put the data inside the bucket by following the structure here:

Right inside the bucket you created, create 3 folders named: 'companies', 'generated', and 'stored\_api\_data'

  

Under 'companies' folder, create 3 folders, named exactly as 'Sasol Limited', 'Arabian Centres', 'Tullow Oil'.

  

Create these 3 folders exactly the same name under both the 'generated' and 'stored\_api\_data' folder.

  

The folder 'companies' is where all the companies' PDFs and files you shared with me via Google Drive are located. Please put them exactly in this folder structure here.

  

Under 'companies' folder:

![](https://t9018625102.p.clickup-attachments.com/t9018625102/8d17ff7f-24e5-4b6c-96e1-f374dafba4ed/Screenshot%20from%202025-04-08%2011-53-42.png)

  

Under 'generated' folder:

![](https://t9018625102.p.clickup-attachments.com/t9018625102/d60217b1-7791-4287-ae51-e2c9f8e28f07/Screenshot%20from%202025-04-08%2011-54-36.png)

  

  

Under 'stored\_api\_data' folder:

![](https://t9018625102.p.clickup-attachments.com/t9018625102/b18d3f2b-a3fb-4b54-aacc-70594bf8fc8f/Screenshot%20from%202025-04-08%2011-54-52.png)

  

  

\*\* It is best not to modify, add or delete any of the Sasol, Cenomi and Tullow's files and folders later, for consistency purpose.

  

I'll share with you both the generated and stored\_api\_data content via google drive.

  

## Gmail - Google Cloud setup for news alert

Go to this page, and follow the instruction under 'Enable the API', 'Configure the OAuth consent screen', and 'Authorize credentials for a desktop application':
https://developers.google.com/workspace/gmail/api/quickstart/python

  

Once you get the JSON credentials, put copy the Client ID and Client Secret into the `.env` file

\*\* Remember to `git pull` first, and you'll see these keys names - `GMAIL_CLIENT_ID` , `GMAIL_CLIENT_SECRET` , `GMAIL_SENDER_EMAIL` appear in the `.env.example` file

  

`GMAIL_SENDER_EMAIL` - this is the email adress where the news alert system will use to send OUT alert emails, ie. this is the email address that appears in the 'from' address of the news alert recipient. This should be a @sentientcap.ai domain email address, as GCP is setup for it.

  

\*\* When you first run the news alert by clicking on an news button, the system might trigger you to login for your Gmail / Google permission. Just follow instruction to allow.

  

\*\* In future stages, using formal email services such as Mailgun is necessary. We are using our own domain as the sender for testing purposes only.

  

## Weaviate

Use my invitation email to create account, and join the organisation I created called 'Sentient Capital'

  

Go to the cluster named: credit-report-vector-db

  

Scroll down to get the weaviate api key (use the Admin key), and go to 'Connect' and copy the weaviate URL, and put these two into the .env file.

  

\*\* Do note that this is a sandbox cluster, and its free trial might expire in 2 weeks, might need to put billing details later.

  

## LLMs API

Go to [platform.openai.com](http://platform.openai.com) and get your api key and put it in .env file

  

Same for Gemini API key, get the api key and put it in .env file

  

## CBonds API (optional at testing stage)

Not necessary, I've stored all the bond data to GCS.

  

## Alpha Vantage API (optional at testing stage)

Not necessary at this stage

\- otherwise free is fine if you prefer.

  

## Run local dashboard

Once everything is properly set up, run command `python app.py` , and you can access the dashboard via the local host address with port provided to you in the terminal.

  

## To pull update for the repo

Simply change directory to the Github repository folder at your terminal and enter `git pull`