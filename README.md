---
# **HHA504 || Exploring AI and Analytics with Pre-trained Models in Azure and GCP**
---

#### **🎯** This task is part of my assignment focused on using pre-trained machine learning models in Azure and Google Cloud Platform (GCP). I am learning to utilize cloud-based notebooks to interact with these models, focusing on their [**`speech`**](https://github.com/raqssoriano/HHA504_assignment_ai/tree/main#-pre-trained-speech-model) and [**`vision`**](https://github.com/raqssoriano/HHA504_assignment_ai/edit/main/README.md#-pre-trained-vision-models) capabilities.

---
---

# _*Steps Taken to Complete this Assignment*_
---

# ☞ _**Pre-trained Speech Model**_
## • GCP Speech-to-Text
- (1) I navigated to **`Google Cloud Console`** ➙ accessed **`Vertex AI`** ➙ and then **`Workbench`**
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/6%20-%20Vertex%20AI.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/7%20-%20managed%20notebook%20%3E%20create%20new%20instance.png" width="650" />.

- (2) I created a new instance/notebook in **`Managed Notebooks`** named **`raqs-gcp-ai-hw`** using preconfigured settings.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/8%20-%20creating%20raqs-gcp-ai-hw%20instance.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/9-%20successful%20creation%20of%20my%20instance.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/10%20-%20inside%20the%20new%20instance%20.png" width="650" />.

- (3) Next, I navigated to the newly created notebook **`raqs-gcp-ai-hw`**, clicked **`Open Jupyterlab`**. I opted to run the scripts I will be using to transcribe an uploaded audio file using the **`Google Console`** (see the screenshot below). First, I uploaded an audio file from my local machine.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/11%20-%20going%20to%20the%20console.png" width="650" />.
  
  - Link to the sample audio file I downloaded to my laptop from Kaggle: [Sample Audio for Speech-to-Text Recognition](https://www.kaggle.com/datasets/pavanelisetty/sample-audio-files-for-speech-recognition)

      <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/12%20-%20pip%20install%20-cloudservices.png" width="650" />.

- (4) After uploading an audio file, I used the documentation/scripts from the GCP I learned by utilizing the tutorials offered by GCP tutorials, which I found really helpful since I had never performed this task before. However, I encountered some error messages that took me hours to troubleshoot. Main issue was the **`Cloud Speech-to-Text API`**.
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/13%20-%20error%20message%20before%20checking%20the%20api.png" width="600" />.
    
- (5) To get a better view of the steps I performed to troubleshoot the API issue, see the screenshots below.

  - I ensured that the Speech-to-Text API was renabled and included in the list of enabled APIs and services.
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/14%20-%20checking%20for%20cloud%20speech%20to%20text%20api.png" width="650" />.
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/15%20-%20api%20enabled%20.png" width="650" />.

  - Even though it was enabled, I was still unsuccessful in running the script I copied and modified from the GCP tutorials. So, I decided to create a new API key.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/16%20-%20api%20credentials.png" width="650" />.
 
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/20%20-%20creating%20new%20api%20key.png" width="650" />.
 
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/21%20-%20api%20key%20created.png" width="650" />.
  
  - After multiple attempts, I successfully ran the scripts to transcribe the uploaded audio file. Below is the script I ran in the **`Jupyterlab`** notebook in GCP to successfully complete this. In my view, creating a new API key was the key that made this happen.
 
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/22%20-%20%20scripts%20for%20speech-to-text.png" width="650" />.
 
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/23%20-%20%20scripts%20for%20speech-to-text.png" width="650" />.

---
--- 

# ☞ _**Pre-trained Vision Models**_

## • GCP Vision API

- (1) I navigated to **`Google Cloud Console`** ➙ searched for **`Vision AI API`** in the search bar ➙ ensured the APIs are enabled. See the screenshots below.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/1.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/2.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/3.png" width="650" />.

- (2) In the same notebook I created for the speech-to-text task, I accessed the **`raqs-gcp-ai-hw`** notebook ➙ opened the **`Jupyterlab`** notebook ➙ uploaded a sample image from my local machine ➙ and installed the **`google cloud vision`** package using pip.
  
    Please click on the [**sample image used for object detection**](https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/softserve.jpeg) to complete this task.
  
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/4.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/5.png" width="650" />.

- (3) Initially, I was having issues running the script I copied from the Google Cloud tutorials, which I modified to match my needs. To succesfully run my script, I had to **add roles** and enable the **Cloud Vision API**. See the screenshots below.
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/6.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/7.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/8.png" width="650" />.

- (4) After multiple attempts at troubleshooting, I successfully ran the script and achieved the desired results. The image I used was very simple and straightforward, and I must say the detected object was accurate..
  
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/9.png" width="650" />.



