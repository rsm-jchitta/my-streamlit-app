import os
import streamlit as st
import boto3
import replicate
import random
import string
import hashlib
import requests
import time
import zipfile
import io
import threading
from dotenv import load_dotenv
from openai import OpenAI
import sendgrid
import time
import re
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Load API keys
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
# Initialize AWS S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("Photography Agent - AI Image Generator")
username_replicate = "rsm-jchitta"
from sendgrid.helpers.mail import Mail

# ✅ Ensure new training state is properly initialized
if "new_training" not in st.session_state:
    st.session_state["new_training"] = False
def replace_names_with_codewords(prompt, name_codeword_dict):
                        """Replace occurrences of person names with their respective codewords."""
                        sorted_names = sorted(name_codeword_dict.keys(), key=len, reverse=True)
                        for name in sorted_names:
                            prompt = re.sub(r'\b{}\b'.format(re.escape(name)), name_codeword_dict[name], prompt, flags=re.IGNORECASE)
                        return prompt
                    
# ✅ 1. Function to Send Email
def send_email(to_email, subject, content):
    """Sends an email notification using SendGrid."""

    if not SENDGRID_API_KEY:
        print("❌ SENDGRID_API_KEY is missing!")
        return

    sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
    email_message = Mail(
        from_email="akhilchitta499@gmail.com",  # Replace with your verified SendGrid email
        to_emails=to_email,
        subject=subject,
        plain_text_content=content
    )

    try:
        response = sg.send(email_message)
        print(f"✅ Email sent! Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")
        
def check_training_status(training_id, training_folder, user_email):
    """Checks the status of a Replicate training session and stores results in S3."""

    if not training_id:
        st.error("❌ Training ID is missing! Unable to check status.")
        return

    st.info(f"📡 Monitoring training status for ID: {training_id}...")

    while True:
        try:
            # ✅ Directly get training status
            training = replicate.trainings.get(training_id)
            status = training.status  # ✅ This directly returns a string like "processing", "succeeded", etc.

            st.write(f"🔍 Current Training Status: {status}")

            if status == "succeeded":
                # ✅ Training completed successfully
                if training.output and "version" in training.output:
                    trained_model_id = training.output["version"]
                    model_req = trained_model_id

                    # ✅ Store model in S3 (✅ Fixed missing `/` in key)
                    try:
                        s3_client.put_object(
                            Bucket=S3_BUCKET_NAME,
                            Key=f"{training_folder}model_req.txt",  
                            Body=model_req.encode("utf-8"),
                            ContentType="text/plain",
                        )
                        st.success(f"✅ Model saved in S3: {model_req}")
                    except Exception as e:
                        st.error(f"❌ Failed to save model in S3: {str(e)}")

                    # ✅ Send Email Notification
                    if user_email:
                        email_subject = "🎉 Training Complete!"
                        email_content = (
                            f"Your training for {training_folder} is complete!\n\n"
                            f"You can now access it under 'Existing Trainings'.\n"
                            f"Model ID: {trained_model_id}"
                        )
                        send_email(user_email, email_subject, email_content)

                else:
                    st.error("❌ Training completed, but no output weights found.")

                break  # ✅ Stop checking after success

            elif status in ["failed", "canceled"]:
                st.error(f"❌ Training {status.capitalize()}! Check Replicate logs for details.")

                # ✅ Send failure email
                if user_email:
                    send_email(
                        user_email,
                        f"❌ Training {status.capitalize()}",
                        f"Your training {training_folder} has {status}. Please check Replicate for details."
                    )
                break  # ✅ Stop checking

            elif status in ["starting", "processing"]:
                time.sleep(30)  # ✅ Check status every 30 seconds

            else:
                st.error(f"❌ Unknown status: {status}. Stopping monitoring.")
                break  # ✅ Stop checking

        except replicate.exceptions.ReplicateError as e:
            st.error(f"❌ Error while checking training status: {str(e)}")
            time.sleep(30)  # ✅ Retry after 30 seconds in case of API failure


# Helper function to check if a folder exists in S3
def folder_exists(bucket, prefix):
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return "Contents" in response

# **New or Existing User Selection**
user_type = st.radio("Are you a new or existing user?", ["New User", "Existing User"])

# **Existing User Flow**
if user_type == "Existing User":
    username = st.text_input("Enter your username to access existing trainings")

    if username:
        user_folder = f"{username}/"

        if folder_exists(S3_BUCKET_NAME, user_folder):
            # Fetch existing training names from S3
            response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=user_folder, Delimiter="/")
            training_list = [prefix["Prefix"].split("/")[-2] for prefix in response.get("CommonPrefixes", [])]

            if training_list:
                selected_training = st.selectbox("Select a training session:", training_list)

                # ✅ Ensure that "New Training" appears ONLY when NO training is selected
                if not selected_training and not st.session_state.get("new_training", False):
                    if st.button("New Training", key="new_training_existing"):
                        st.session_state["new_training"] = True  # ✅ No refresh needed


                # ✅ If a training is selected, load the model
                if selected_training:
                    model_req_key = f"{user_folder}{selected_training}/model_req.txt"
                    trained_model_id = None
                    

                # ✅ Prevent checking `model_req.txt` if new training is happening
                    if not st.session_state.get("new_training", False):  
                        try:
                            model_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=model_req_key)
                            trained_model_id = model_obj["Body"].read().decode("utf-8")
                            st.success(f"✅ Model Loaded: {trained_model_id}")
                        except s3_client.exceptions.NoSuchKey:
                            st.warning("⚠ Model training has not started yet. Please upload images and start training.")
                            trained_model_id = None
                        except Exception as e:
                            st.error(f"❌ Error loading model: {str(e)}")
                            trained_model_id = None


                    # ✅ Prompt input and image generation
                    
                  

                    # ✅ Load name-codeword mapping from S3
                    name_codeword_map_key = f"{user_folder}{selected_training}/name_codeword_map.txt"

                    try:
                        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=name_codeword_map_key)
                        name_codeword_data = obj["Body"].read().decode("utf-8").split("\n")

                        name_codeword_dict = {}
                        for line in name_codeword_data:
                            if ":" in line:
                                name, code = line.split(":")
                                name_codeword_dict[name.strip()] = code.strip()

                    except Exception as e:
                        st.warning("⚠ Name-codeword mapping not found. Using raw prompt.")
                        name_codeword_dict = {}


                    # ✅ Prompt input and image generation
                    prompt = st.text_input("Enter a prompt to generate an image")

                    if st.button("Generate Image"):
                        if not st.session_state.get("generate_image", False):  # ✅ Prevent unnecessary reruns
                            st.session_state["generate_image"] = True
                            st.session_state["prompt_text"] = prompt  # ✅ Store prompt


                    if st.session_state.get("generate_image", False):
                        st.session_state["generate_image"] = False  # ✅ Reset state to avoid infinite loop

                        prompt = st.session_state.get("prompt_text", "")
                        if prompt:
                            modified_prompt = replace_names_with_codewords(prompt, name_codeword_dict)

                            # ✅ Call Replicate API with the modified prompt
                            output = replicate.run(
                                trained_model_id,
                                input={"prompt": modified_prompt},
                            )

                            if isinstance(output, list):
                                image_url = output[0]
                                st.success(f"✅ Image Generated! [Click here to view]({image_url})")
                            else:
                                st.error("❌ Failed to generate image.")

            
elif user_type == "New User":
    if "registered_username" not in st.session_state:
        st.session_state["registered_username"] = None  # Initialize session state

    username = st.text_input("Choose a unique username")

    # ✅ Check if the username is already taken **only if not already stored**
    if username and st.session_state["registered_username"] is None:
        user_folder = f"{username}/"
        if folder_exists(S3_BUCKET_NAME, user_folder):
            st.error(f"❌ The username '{username}' is already taken! Please choose another.")
            st.stop()  # ✅ Prevent further execution if username exists
    
    email = st.text_input("Enter your email for training notifications")

    if email and st.button("Register"):
        user_folder = f"{username}/"

        # ✅ Store the username in session state so it's not checked again later
        st.session_state["registered_username"] = username  

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{user_folder}email.txt",
            Body=email.encode("utf-8"),
            ContentType="text/plain",
        )

        st.success(f"✅ Account '{username}' created! Proceed with training.")

            
#  Check if the user exists in S3
if "username" in locals() and folder_exists(S3_BUCKET_NAME, f"{username}/"):

    # ✅ Button to start a new training session
    if "new_training" not in st.session_state:
        st.session_state["new_training"] = False

    if st.button("New Training", key="new_training_new_user"):
        st.session_state["new_training"] = True

    # ✅ If user clicked "New Training", show input fields
    if st.session_state["new_training"]:
        st.subheader("📌 Start a New Training Session")

        training_name = st.text_input("Enter training name")

        if st.button("Cancel New Training", key="cancel_new_training"):
            st.session_state["new_training"] = False  # ✅ Reset without refreshing


        if training_name:
            training_folder = f"{username}/{training_name}/"

            # Check if the folder exists
            if folder_exists(S3_BUCKET_NAME, training_folder):
                if not st.session_state.get("training_created", False):
                    st.warning("⚠ Training folder already exists. Continuing with training setup...")
            else:
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=f"{training_folder}placeholder.txt",
                    Body="Placeholder file".encode("utf-8"),
                    ContentType="text/plain",
                )
                st.session_state["training_created"] = True
                

            
          # Replace with your Replicate username
            import re

            # ✅ Function to sanitize the training name for Replicate
            def sanitize_model_name(name):
                name = name.lower().strip()  # Convert to lowercase & remove leading/trailing spaces
                name = re.sub(r'[^a-z0-9._-]', '-', name)  # Replace invalid characters with '-'
                name = re.sub(r'[-_.]+$', '', name)  # Remove trailing special characters
                name = re.sub(r'^[-_.]+', '', name)  # Remove leading special characters
                return name
            import hashlib
            import time

            # ✅ Generate a unique model name by hashing (Username + Training Name + Timestamp)
            def generate_unique_model_name(username, training_name):
                unique_string = f"{username}_{training_name}_{int(time.time())}"  # Combine with timestamp
                hashed_name = hashlib.md5(unique_string.encode()).hexdigest()[:8]  # Generate a short hash
                return sanitize_model_name(f"{username}-{training_name}-{hashed_name}")

            MODEL_NAME = generate_unique_model_name(username, training_name)

            
            if not MODEL_NAME:  # Check if sanitization results in an empty name
                st.error("❌ Invalid training name! Please enter a valid name.")
                st.stop()  # Prevent further execution if the name is invalid
            # ✅ Store the model name in session state once created
            # ✅ Store the model name in session state once created
            if "created_model_name" not in st.session_state:
                st.session_state["created_model_name"] = None
            if "model_already_exists" not in st.session_state:
                st.session_state["model_already_exists"] = False

            if training_name:
                import datetime
                import re

                # ✅ Generate a unique model name using username + current timestamp
                def generate_unique_model_name(username):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # YYYYMMDDHHMMSS
                    model_name = f"{username}-{timestamp}".lower()  # Convert to lowercase

                    # ✅ Remove invalid characters (anything except a-z, 0-9, -, _, .)
                    model_name = re.sub(r'[^a-z0-9._-]', '-', model_name)

                    # ✅ Ensure it doesn't start or end with -, _, .
                    model_name = re.sub(r'^[-_.]+|[-_.]+$', '', model_name)

                    return model_name

                # ✅ Generate and sanitize model name
                MODEL_NAME = generate_unique_model_name(username)

                # ✅ Create the model directly
                model = replicate.models.create(
                    owner=username_replicate,
                    name=MODEL_NAME,
                    hardware="gpu-t4",
                    visibility="public",
                    description="A fine-tuned FLUX.1 model trained on custom images"
                )

                # ✅ Store the created model in session state
                st.session_state["created_model_name"] = MODEL_NAME  
                st.success(f"🎉 Model '{MODEL_NAME}' created successfully!")



            # ✅ Step 3: Proceed with training            

            # Move forward with person name input
            person1 = st.text_input("Enter name of Person 1")
            person2 = st.text_input("Enter name of Person 2")

            if person1 and person2:
                # ✅ Generate 4-letter unique codes
                def generate_codeword(name):
                    seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % (10**8)
                    random.seed(seed)
                    return name[:2].lower() + ''.join(random.choices(string.ascii_lowercase, k=2))

                code_word_for_person1 = generate_codeword(person1)
                code_word_for_person2 = generate_codeword(person2)

                # ✅ Create folders for person1, person2, and combined images
                person1_folder = f"{training_folder}{code_word_for_person1}/"
                person2_folder = f"{training_folder}{code_word_for_person2}/"
                combined_folder = f"{training_folder}{code_word_for_person1}_{code_word_for_person2}/"

                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=f"{person1_folder}placeholder.txt", Body="".encode("utf-8"))
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=f"{person2_folder}placeholder.txt", Body="".encode("utf-8"))
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=f"{combined_folder}placeholder.txt", Body="".encode("utf-8"))
                name_codeword_map = f"{person1}:{code_word_for_person1}\n{person2}:{code_word_for_person2}"

                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=f"{training_folder}name_codeword_map.txt",  # ✅ Save in the training folder
                    Body=name_codeword_map.encode("utf-8"),
                    ContentType="text/plain",
                )
                st.success("✅ Subfolders created for each person and combination!")

                # ✅ Image Upload Section
                st.info("Upload 5 pictures for each category.")

                person1_images = st.file_uploader("Upload 5 images of Person 1", accept_multiple_files=True, key="p1")
                person2_images = st.file_uploader("Upload 5 images of Person 2", accept_multiple_files=True, key="p2")
                combined_images = st.file_uploader("Upload 5 images of both together", accept_multiple_files=True, key="combo")
                # ✅ Store name → codeword mapping in a text file
                


               # ✅ Step 1: Show Captions for Editing BEFORE Training Starts
                # ✅ Step 1: Upload Images Before Generating Captions
                st.info("📤 Uploading images to S3...")

                for image, folder in zip(
                    [person1_images, person2_images, combined_images],
                    [code_word_for_person1, code_word_for_person2, f"{code_word_for_person1}_{code_word_for_person2}"]
                ):
                    s3_folder = f"{training_folder}{folder}/"
                    
                    for img in image:
                        s3_key = f"{s3_folder}{img.name}"

                        # ✅ Upload image to S3 before generating captions
                        s3_client.upload_fileobj(img, S3_BUCKET_NAME, s3_key, ExtraArgs={'ACL': 'public-read'})

                st.success("✅ All images uploaded successfully!")

                # ✅ Step 2: Generate Captions Only After Images Are Uploaded
                st.subheader("📌 Review & Edit Captions Before Training")
                edited_captions = {}

                for image, folder in zip(
                    [person1_images, person2_images, combined_images],
                    [code_word_for_person1, code_word_for_person2, f"{code_word_for_person1}_{code_word_for_person2}"]
                ):
                    s3_folder = f"{training_folder}{folder}/"

                    for img in image:
                        s3_key = f"{s3_folder}{img.name}"

                        # ✅ Generate pre-signed URL for OpenAI Vision API
                        image_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
                            ExpiresIn=300  # ✅ Temporary URL (valid for 5 minutes)
                        )

                        # ✅ Generate a caption using OpenAI Vision API
                        response_vision = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Describe this image in one sentence(atleast 10 words mandatory). Always start your sentence with 'A photo of {folder.replace('_', ' and ')}' even if you cant assist."},
                                        {"type": "image_url", "image_url": {"url": image_url}},
                                    ],
                                }
                            ],
                            max_tokens=300,
                        )

                        # ✅ Extract AI-generated caption
                        generated_caption = response_vision.choices[0].message.content

                        # ✅ Show editable text area
                        edited_caption = st.text_area(f"Edit caption for {img.name}:", value=generated_caption, key=img.name)

                        # ✅ Store edited caption
                        edited_captions[img.name] = edited_caption


                        # ✅ Generate a caption using OpenAI Vision API
                        response_vision = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Describe this image in one sentence(atleast 10 words mandatory). Always start your sentence with 'A photo of {folder.replace('_', ' and ')}' even if you cant assist."},
                                        {"type": "image_url", "image_url": {"url": image_url}},
                                    ],
                                }
                            ],
                            max_tokens=300,
                        )

                        # Extract AI-generated caption
                        generated_caption = response_vision.choices[0].message.content

                        # ✅ Show editable text area
                        edited_caption = st.text_area(f"Edit caption for {img.name}:", value=generated_caption, key=img.name)

                        # ✅ Store edited caption
                        edited_captions[img.name] = edited_caption

                # ✅ Step 2: Require User to Confirm Caption Edits Before Proceeding
                # ✅ Step 3: Require User to Confirm Caption Edits Before Training
                if st.button("Save Captions & Proceed to Training"):
                    for img_name, final_caption in edited_captions.items():
                        txt_file_key = f"{training_folder}{img_name.rsplit('.', 1)[0]}.txt"

                        # ✅ Save edited captions in S3
                        s3_client.put_object(
                            Bucket=S3_BUCKET_NAME,
                            Key=txt_file_key,
                            Body=final_caption.encode("utf-8"),
                            ContentType="text/plain",
                        )

                    st.success("✅ Captions saved! Now you can start training.")
                    st.session_state["captions_saved"] = True  # ✅ Store in session

                # ✅ Step 4: Only Show "Start Training" Button After Captions Are Saved
                if st.session_state.get("captions_saved", False):
                    if st.button("Start Training"):
                        if len(person1_images) < 5 or len(person2_images) < 5 or len(combined_images) < 5:
                            st.error("❌ Please upload 5 images for each category before starting training.")
                        else:
                            st.session_state["training_started"] = True  # ✅ Training starts only now
 # ✅ Training starts only now

                # ✅ Set training state only once

                if st.session_state.get("training_started", False):  # ✅ Now training will continue without rerunning
                    st.info("🚀 Training has started... Please wait.")
                    st.session_state["captions_saved"] = False

                    if len(person1_images) >= 5 and len(person2_images) >= 5 and len(combined_images) >= 5:
                        for image, folder in zip(
                            [person1_images, person2_images, combined_images],
                            [code_word_for_person1, code_word_for_person2, f"{code_word_for_person1}_{code_word_for_person2}"]
                        ):
                            s3_folder = f"{training_folder}{folder}/"
                            for img in image:
                                s3_key = f"{s3_folder}{img.name}"
                                s3_client.upload_fileobj(img, S3_BUCKET_NAME, s3_key)

                                #  Generate pre-signed URL for OpenAI Vision API
                                image_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

                                #  Generate a caption using OpenAI Vision API
                                response_vision = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": f"Describe this image in one sentence(atleast 10 words mandatory). Always start your sentence with 'A photo of {folder.replace('_', ' and ')}' even if you cant assist."},
                                                {"type": "image_url", "image_url": {"url": image_url}},
                                            ],
                                        }
                                    ],
                                    max_tokens=300,
                                )

                                # Extract the AI-generated caption
                                caption = response_vision.choices[0].message.content

                                # Generate text file name (same as image but with `.txt` extension)
                                txt_file_key = f"{s3_folder}{img.name.rsplit('.', 1)[0]}.txt"

                                #  Upload the caption as a text file to S3
                                s3_client.put_object(
                                    Bucket=S3_BUCKET_NAME,
                                    Key=txt_file_key,
                                    Body=caption.encode("utf-8"),
                                    ContentType="text/plain",
                                )

                        st.success("Images uploaded successfully!")
                        st.info("Creating a ZIP file for training...")
                        # ✅ 1. Create ZIP File in Memory
                        s3_folder_path = f"{training_folder}"
                        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_folder_path)

                        if 'Contents' in response:
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for obj in response['Contents']:
                                    file_key = obj['Key']

                                    if file_key.endswith('/') or "placeholder.txt" in file_key:  # ✅ Skip folders and placeholder
                                        continue  

                                    file_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
                                    file_content = file_obj['Body'].read()

                                    # ✅ Keep the folder structure inside the ZIP
                                    filename_with_path = file_key.replace(training_folder, "", 1)  # Remove base training path
                                    zip_file.writestr(filename_with_path, file_content)

                            zip_buffer.seek(0)
                            st.success("✅ ZIP file created including all subfolders!")
                        else:
                            st.error(f"❌ No files found in {s3_folder_path}. Cannot create ZIP.")

                        
                        # ✅ Retrieve user email from S3
                        email_key = f"{username}/email.txt"
                        try:
                            email_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=email_key)
                            user_email = email_obj["Body"].read().decode("utf-8")
                        except Exception as e:
                            user_email = None
                            st.warning("⚠ Email not found. No notification will be sent.")
                        st.info("Starting model training on Replicate...")

                        # ✅ 2. Stream ZIP Directly to Replicate API
                        with zip_buffer as zip_file:
                            training = replicate.trainings.create(
                                version="ostris/flux-dev-lora-trainer:4ffd32160efd92e956d39c5338a9b8fbafca58e03f791f6d8011f3e20e8ea6fa",
                                input={
                                    "input_images": zip_file,  # ✅ Stream the ZIP file directly
                                    "steps": 2000,
                                    "batch_size": 1,
                                    "lora_rank": 32,
                                    "optimizer": "adamw8bit",
                                    "learning_rate": 0.0001,
                                    "autocaption": False,
                                    "caption_dropout_rate": 0,
                                },
                                destination=f"{username_replicate}/{MODEL_NAME}",
                            )

                        st.success("📩 Training has started. You will receive an email once it's completed.")

                        st.info("🔄 After receiving the email, go to 'Existing Trainings' to use your model.")
                        
                        # ✅ Immediately start monitoring the training process
                        if training and hasattr(training, "id"):
                            training_id = training.id  # ✅ Get the training ID from Replicate
                            st.success(f"📡 Training started! Monitoring status for training ID: {training_id}...")

                            # ✅ Call function to monitor training status
                            check_training_status(training_id, training_folder, user_email)
                        else:
                            st.error("❌ Training failed to start. Please check Replicate logs for details.")

                        
                        

