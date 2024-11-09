---
# **HHA504 || Exploring AI and Analytics with Pre-trained Models in Azure and GCP**
---

#### **🎯** This task is part of my assignment focused on using pre-trained machine learning models in Azure and Google Cloud Platform (GCP). I am learning to utilize cloud-based notebooks to interact with these models, focusing on their [**`speech recognition`**](https://github.com/raqssoriano/HHA504_assignment_ai/tree/main#-pre-trained-speech-model), [**`GCP vision`**](https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/README.md#-gcp-vision-api), and [**`Azure vision`**](https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/README.md#-azure-ai-vision) capabilities.

#### **📌** The [Comparison between GCP and Azure](https://github.com/raqssoriano/HHA504_assignment_ai/edit/main/README.md#-gcp-vs-azure-and-my-reflections-working-on-these-platforms-in-speech--vision-models-on-these-platforms) and my [final thoughts](https://github.com/raqssoriano/HHA504_assignment_ai/edit/main/README.md#-my-final-thoughts-) on using these two platforms to complete this assignment.

---
---

# _*Steps Taken to Complete this Assignment*_
---

# ☞ _**Pre-trained Speech Model:**_
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
  
  - Link to the sample audio file I downloaded to my laptop from Kaggle: [Kaggle Audio](https://www.kaggle.com/datasets/pavanelisetty/sample-audio-files-for-speech-recognition) or the [Sample Audio for Speech-to-Text Recognition](https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20speech%20to%20text/sampleaudio.wav)

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

  - In an attempt to complete this task, I used [**`Google Cloud Speech-to-Text tutorials`**](https://cloud.google.com/speech-to-text?hl=en#transcribe-audio).

---
--- 

# ☞ _**Pre-trained Vision Models:**_

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

- (4) After multiple attempts at troubleshooting, I successfully ran the script and achieved the desired results. The image I used was very simple and straightforward, and I must say the detected object was accurate. In an attempt to complete this task, I used the [**`Google Cloud Vision API tutorials`**](https://cloud.google.com/vision/docs/how-to).
  
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/GCP%20Vision%20API/9.png" width="650" />.


## • Azure AI Vision

- (1) I navigated to **`Azure portal`** ➙ searched for **`AI Machine Learning Studio`** in the search bar ➙ inside the AI machine learning studio, I created a new workspace named **`raqs-azure-ai-hw`** and used the default configuration settings.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/1.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/2.png" width="500" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/3.png" width="650" />.

- (2) I created a new compute instance inside the newly created workspace named **`raqs-azure-ai-hw`**.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/4.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/5.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/6.png" width="650" />.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/7.png" width="650" />.

- (3) Additionally, within the workspace, I created a new file inside a notebook named **`raqs-azure-ai-hw.ipynb`** and uploaded the [sample image](https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/softserve.jpeg). I used [**`Azure online tutorials`**](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?view=azureml-api-2) that I thought might be helpful in an attempt to complete this task.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/8.png" width="650" />.
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/9.png" width="650" />.
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/10.png" width="650" />.
    
    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/11.png" width="650" />.
  
- (4) I ran scripts several times a few days ago, which I copied from the [**`Azure machine learning tutorial`**](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-azure-ml-in-a-day?view=azureml-api-2) in an attempt to complete this task. However, I was unsuccessful and gave up trying since I was not sure if the process I have taken was incorrect.

    <img src="https://github.com/raqssoriano/HHA504_assignment_ai/blob/main/Azure%20AI%20Vision/12.png" width="650" />.

---

# ☞ _**GCP vs. Azure and My Reflections Working on these Platforms in Speech & Vision Models on These Platforms**_

---

<table>
<thead>
<tr>
<th> </th>
<th>Speech Recognition</th>
<th>Vision Capabilities</th>
</tr>
</thead>
<tbody>
<tr>
<td>➤GCP</td>
    
<td>
  (1) Accuracy in Transcribing the Uploaded Audio File: The result or transcript was accurate. However, I used a simple audio file, which could be why. I wonder if it would be the same case if there were background noise, different accents, varying speed, or complex termnilogies. This would be interesting to find out.<br> <br>
  (2) APIs and Other Services: Through Google Cloud tutorials, I was able to follow through the steps, which seemed pretty straightforward. However, when attempting to run scripts to transcribe the uploaded audio file, I had to perform multiple troubleshooting steps. In the end, creating an API key helped resolve the issue, and I was able to successfully perform speech recognition. The default configuration settings when starting a new notebook and compute instance made it quite easy to perform this task. Also, locating the documentation/scripts to be used for speech recognition through the tutorials was easier compared to Azure. Basically, the only difficulty that I experienced was getting stuck due to the API key errors.<br> <br>
  
  (3) User-friendly Interface: In completing this task, it gave me an option to run scripts either in notebook or the google cloud console which I found it very convenient.<br> <br>
  (4) I found the experience of performing speech-to-text recognition process on GCP quite straightforward. Although I encountered errors while running the script to detect an audio file, I was able to resolve them by creating an API key.<br> </td>

<td>
  ➤ GCP Vision API:<br> <br>
    
  (1) The setup process was pretty straightforward with its simple and comprehensive documentation found in tutorials, which I found it very helpful in performing an oject/image detection.<br> 

  (2) Similar to my experience with speech recognition, I used a very simple sample image file which I uploaded to GCP which resulted in straightforward detection, so that could be why it was almost accurate. One of the issues I encountered was choosing what could be the correct script to match the uploaded image, as there were various detection scripts (like detect image properties/logos/faces/landmarks/text in images etc.) listed in the Google Cloud Vision API documentation, which I modified. However, the how-to-guides in GCP were definitely better and less confusing compared to Azure.<br> 
  
  (3) Another issue I encountered after getting the object detection result was understanding the meaning of the "`confidence: 0.67`" result for the uploaded image. I discovered thru this [Google Cloud Community Blog link](https://www.googlecloudcommunity.com/gc/Community-Blogs/Unlocking-visual-intelligence-with-Google-Cloud-s-Vision-API/ba-p/682623) that it signifies the confidence score, which ranges from 0 (no confidence) to 1 (very high confidence).
  
  (4) Troubleshooting the errors I encountered in the GCP Vision API while completing this assignment was easier to manage compared to Azure. As mentioned previously, creating an API key helped resolve the issues. Also, I ensured that the correct roles and permissions were set up.<br> </td>

</tr>
<tr>
<td>➤Azure</td>


    
<td>          N/A         <br> </td>

<td>
  ➤ Azure AI Vision:<br> <br>
    
  (1) Similar to GCP, Azure provides tutorials with documentation. However, I found the guides to be more complex and less user-friendly. I spent hours searching for the correct script to use, but none matched what I needed to successfully detect the simple uploaded image file. I kept ending up at the training script section. Perhaps I didn't look hard enough to locate the correct documentation, or maybe I was searching for the wrong terms.<br> <br>
  (2) Setting up the workspace, compute instance, and notebook was quite easy, similar to GCP, and I didn't encounter any issues.<br>
  
  (3) Obviously, I was not successful in completing this task, so I couldn't state the accuracy or reliability of Azure AI vision as compared to GCP when it comes to image/object detection. However, I can say that troubleshooting in Azure required more effort.<br> </td>

</td>
</tr>

</tr>
</tbody>
</table>

---
---

# ☞ _**My Final Thoughts:**_ 👩🏻‍💻

#### 📌 Based on my experience to this assignment, GCP has stood out for its user-friendly setup and comprehensive documentation, making it easier to navigate tutorials and find solutions for successfully running scripts. Despite occasional issues, the troubleshooting process in GCP was more manageable, delivering almost accurate image detection. However, with Azure's guides/tutorials, finding the correct documentation was more challenging and time-consuming. I learned from the class zoom recording, mentioned by my professor, that one of the steps could be using the SDK for image analysis, which I only discovered after completing the Azure AI vision section of this assignment. I still intend to follow the link shared by my professor, but decided to submit this documentation process as part of my learning curve. I believe if I had been successful in the Azure part, it would be a great platform for detecting objects or images. For this homework, GCP is my preferred choice for performing speech-to-text and object/image recognition because of the less complicated process I experienced and the successful completion of this assignment.


