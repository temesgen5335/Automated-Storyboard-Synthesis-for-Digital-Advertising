# 10 Academy Cohort B — Week 12 (original brief)

> Extracted verbatim from the provided .docx for durable context.

10 Academy Cohort B
Weekly Challenge: Week 12
Semantic Image and Text Alignment: Automated Storyboard Synthesis for Digital Advertising

Business objective  
Recent advancements in machine learning, natural language processing, and computer vision, alongside the development of Large Language Models (LLMs), have ushered in a new era of capabilities in the digital domain. These technologies enable the intricate processing and interpretation of data, facilitating the creation of detailed, dynamic content that bridges the gap between textual concepts and visual storytelling. The integration of these technologies not only simplifies the translation of complex ideas into tangible visuals but also enhances creativity and efficiency in content generation. The business objective of the challenge is to harness these capabilities to transform textual descriptions of advertisement concepts and assets into detailed storyboards. This transformation process aims to visually depict the narrative flow and user interaction within advertisements, making the conceptualization of digital campaigns both more intuitive and impactful.

Adludio is at the forefront of online mobile advertising, specialising in the creation of interactive ads that resonate with viewers through dynamic content such as mini-games, videos, texts, and images. Adludio offers its clients a suite of services designed to maximise engagement and campaign performance:
Collection of comprehensive briefs detailing brand identity, advertising objectives, guidelines, KPIs, objectives, and budget.
Design of interactive advertisements, leveraging a rich media creative toolkit.
Distribution of these creatives to targeted audiences via sophisticated real-time bidding for impressions on the open market.
Optimization of the creative design and targeting process through advanced machine learning algorithms to ensure maximum impact.

In this transformative era of advertising and recognizing the potential for technology to streamline and enhance the ad creation process, Adludio is embarking on an ambitious initiative to automate the end-to-end process of advertising production. This automation aims to significantly expedite the ideation and execution phases, enabling clients to swiftly launch their campaigns with minimal expenditure of time and resources. A key component of this automation involves the generation of potential creative concepts based on the client's brief. By leveraging advanced machine learning algorithms, Adludio intends to present clients with viable creative options rapidly, thereby reducing the traditional turnaround time from over a week to mere days. Your task, as part of this transformative process, is to architect and develop a cutting-edge machine learning solution that automates the conversion of textual advertisement concepts, assets descriptions into visually compelling storyboards. This solution should intelligently interpret the provided concepts and assets, generate relevant visual and textual assets, compose these assets into individual ad frames, and ultimately synthesize a cohesive storyboard that encapsulates the essence of the proposed ad campaign.



Background & Context

Start by reading these two blogs for a general understanding on the context and background of this challenge:  
Dynamic Creative & Content Optimization - DCO Marketing & Advertising (claravine.com)
5 Examples That Show How Machine Learning is Changing Digital Advertising
How Machine Learning Is Shaping The Future Of Advertising (forbes.com)

You may find this part helpful in understanding some of the technical terms you may find in the datasets.
Creative - an advertisement (ad) that users encounter and interact with while navigating a website or utilising a mobile application powered by ads.
Concept - envisioned appearance and structure of the advertisement, outlining the creative idea behind it.
AdFrame - a creative is composed of several segments or scenes, each of which is called an AdFrame. These frames collectively tell the story or convey the message of the advertisement.
AdFormat - specifies the dimensions (width and height) of the space where the advertisement will be displayed. Common formats include Full Screen (FS) with dimensions of 320x480 pixels and Mid-Page Unit (MPU) with dimensions of 300x250 pixels.

The below diagram shows the flow of an event after we won an impression & an ad is displayed to the publisher.  Our Ads have multiple interactive screens animated with user interaction. Depending on the type of creative, an ad may have one or more screens. See the diagram below to understand how the user interacts with our ads.
		Fig 1: Example of a user interacting with an ad ( Event flow). 161925209550

Here are some examples of our latest ads:
Indica, ITC, Detran
Data

You will receive access to the following datasets, accompanied by a comprehensive breakdown of their contents and organizational structure:
Archive Folder: This archive features an 'Assets' folder, which, as the name implies, contains the images used to construct the creatives. Within this folder, subfolders correspond to different creative projects, each containing various assets integral to the creative. Notably, two crucial images, labeled 'landing' and 'endframe,' serve as the initial and final frames of the advertisement, respectively. These images, along with all other assets required to assemble these keyframes, are included within the respective subfolders. [LINK]
Sample Concepts with Assets and Size Descriptions. A JSON file that outlines a series of concepts. Each object within the file includes the following details:
Concept: The creative idea's name.
Implementation: A detailed, frame-by-frame breakdown, providing visual representations and explanations for each segment.
Explanation: An overarching description of the advertisement's concept and the intended user flow.
Asset-Suggestions: For each frame, a curated list of three recommended assets is provided, detailing the category and a brief description of each suggested element.
Category description here
Storyboard Examples: To offer insights into the standard of deliverables Adludio presents to its clients, a selection of sample storyboards is provided. These examples illustrate various approaches to storyboard composition and design:
Multiple StoryBoard Examples

$ ls Challenge_Data/ 
                           PRE Assets/
$ls Challenge_Data/Assets/
                           PRE 002dbbd85ef3fe6a2e7d0754fb9f9a1a/
                           PRE 00508a6979fdd9c2e5f8c68bfefe5f71/
                           PRE 00dfe88c4d3fb60793765d314bf24b7c/
                           PRE 015efcdd8de3698ffc4dad6dabd6664a/
                           PRE 015f38df736f5a9498b18a1d12170187/

$ls Challenge_Data/Assets/ff88446626571f68e23c4b07063fd806/ 
2022-10-16 19:17:09          0 
2022-10-16 19:17:13       1002 MPU-click-area.png
2022-10-16 19:17:14     220388 _preview.png
2022-10-16 19:17:15       9464 after-copy.png
2022-10-16 19:17:09      40736 after.jpg
2022-10-16 19:17:10      57892 before.jpg
2022-10-16 19:17:16      12005 cta.png
2022-10-16 19:17:11      48477 end.jpg
2022-10-16 19:17:12      37609 endframe.jpg
2022-10-16 19:17:16       1557 engagement_animation_1.png
2022-10-16 19:17:17       1979 engagement_animation_2.png
2022-10-16 19:17:13      59853 landing.jpg
2022-10-16 19:17:17      23328 video-cta.png
2022-10-16 19:17:18    4369490 video.mp4




Learning Outcomes
Skills:
Working with Deep Learning frameworks e.g. pyTorch and Tensorflow
Optimising image segmentation deep learning architectures 
Using CV public models and APIs
ML modelling with KPIs and DL artefacts  
Formulating and designing test and training strategies 
Central Logging systems
MLOps  with DVC, CML, and MLFlow
Knowledge:
Deep Learning algorithms for computer vision  
Machine learning 
Hyperparameter tuning
Model comparison & selection
Experiment analysis
data privacy, data security, ethical use of data. The 8 principles of responsible machine learning
Communication:
Explaining complex subjects


Competency Mapping
The tasks you will carry out in this week’s challenge will contribute differently to the 11 competencies 10 Academy identified as essential for job preparedness in the field of Data Engineering, and Machine Learning engineering. The mapping below shows the change (lift) one can obtain through delivering the highest performance in these tasks.   


Competency
Potential contributions from this week
Professionalism for a global-level job
Articulating business values
Collaboration and Communicating
Reporting  to stakeholders
Software Development Frameworks
Using Github for CI/CD, writing modular codes, and packaging
Python programming
Advanced use of python modules such as Pandas, Matplotlib, Numpy, Scikit-learn, Prophet and other relevant python packages
SQL programming
MySQL db create, read, and write
Data & Analytics Engineering

data filtering, data transformation, and data warehouse management
MLOps & AutoML
Pipeline design, data and model versioning,  
Deep Learning and Machine Learning
NLP, topic modelling, sentiment analysis
Web & Mobile app programming
HTML, CSS ,Flask, Streamlit

Team
Tutors: 
Yabebal
Emtinan
Rehmet
Badges
Each week, one user will be awarded one of the badges below for the best performance in the category below.

In addition to being the badge holder for that badge, each badge winner will get +20 points to the overall score.

Visualization - quality of visualizations, understandability, skimmability, choice of visualization
Quality of code - reliability, maintainability, efficiency, commenting - in future this will be CICD/CML
Innovative approach to analysis -using latest algorithms, adding in research paper content and other innovative approaches
Writing and presentation - clarity of written outputs, clarity of slides, overall production value
Most supportive in the community - helping others, adding links, tutoring those struggling

The goal of this approach is to support and reward expertise in different parts of the Machine learning engineering toolbox.
Group Work Policy
This submission can be done either in a group or individually. You should let us know your choice using the following google document.


Late Policy
Our goal is to prepare successful learners for a global level job. At work, deadlines are sometimes very strict - either you do it before the deadline or the company loses a substantial opportunity.  Moreover, the late communication behaviour (submission in 10 Academy can be considered as progress communication to team leads), blinds team leads and CEOs and is very determinantal in hindering the success of the company.
We have set our late submission as follows
Submissions are accepted only within the 12 hrs window - 17:00 UTC - 7:00 UTC  of the submission deadline
Frequently late submissions (exceeding 6 total late submissions) will disqualify a person from the list of trainees 10 Academy recommends to partner employers.
Badges will be rewarded for the cumulative on-time appearances (gmeet calls, on-time assignment submissions, and other places where being on-time is important) 
From week 8 onwards, your two lowest weeks’ scores will not be considered.





















Instruction:

In the evolving landscape of digital advertising, the capability to automatically transform textual descriptions of advertisements into visual storyboards represents a significant leap towards creativity and efficiency. 

This challenge aims to leverage the latest advancements in 
machine learning, 
image, and text generation technologies 
LLM based agents
to automate the storyboard creation process. By providing you with textual inputs detailing the concept, assets, and size of an advertisement, this initiative seeks to explore the potential of AI in streamlining the design process, enhancing ad engagement, and optimising campaign performance. 

The ultimate objective is to develop a machine learning framework that can seamlessly convert textual ad descriptions into detailed, visually compelling storyboards that accurately reflect the intended user flow and narrative of an advertising campaign.

To systematically address the challenge, we have divided the problem into a set of 3 consecutive tasks, each building upon the insights and outputs of the previous one. This structured approach is designed to guide you through the process of transforming textual inputs into a comprehensive storyboard that visually narrates the ad campaign's flow.
Task 1: EDA & Workflow Strategy 
Review resources
Perform EDA on the given data 
Understand the data provided
Prepare your environment for asset analysis
Identify ML and DL models to help you analyse the data, label items in image,  segment components from image, etc.
YOLO 
Scikit-learn
UNET++ 
Task 2: Critic/Grading Agent Asset Analysis and Automatic Asset Editing with AutoGen as an agent. 
Understanding and manipulating the creative assets provided for the advertisement storyboards using AutoGen agents. These agents will represent different roles in the application.
Creating a compelling storyboard requires innovative solutions for understanding creative assets and auto-editing and when necessary generating images and text. 
Your primary task here is to use LLM, CV, and ML models to get a good understanding of creative assets and to be able to edit or generate creative assets such as images and texts.
References to check:
[Advanced] Build Agents with Vision Abilities Using OpenAI & AutoGen & Llava & Stable Diffusion (henrywithu.com)
AI Agents for Data Visualization with AutoGen (mlq.ai)
Multi-Agent AutoGen with Functions - Step-by-Step with Code Examples | by Dr. Ernesto Lee | Medium (drlee.io)
autogen/notebook at main · microsoft/autogen (github.com)
Understanding Creative Assets

Exploring Asset Data:
Begin by thoroughly exploring the dataset provided, which includes images, textual descriptions, and JSON files with concept breakdowns.
Understand the structure of the dataset, identifying key elements such as ‘landing’ and ‘endframe’ images, as well as other assets within the subfolders.
Utilizing AutoGen Agents for Asset Analysis:
Use different AutoGen agents to analyze the images and text. Each agent will perform specific tasks to extract meaningful insights and patterns from the data.
Agents and Their Tasks
Image Analysis Skill:
Object Identification: Identify key objects within the images and understand their relevance to the advertisement.
Color Identification: Extract the primary colors used in the images to maintain visual consistency.
Position Extraction: Determine the position of objects within the images to help in composing ad frames.
Character Recognition: Use OCR (Optical Character Recognition) techniques to extract text from images.
Text Analysis Skill:
Text Summarization: Summarize text descriptions to capture key concepts and narrative flow.
Key Phrase Identification: Identify important phrases and terms that define the advertisement’s message and tone.
Narrative Understanding: Understand the overall narrative to ensure alignment with visual elements.
The agents must have the knowledge base that can be used to grade and provide feedback for the image composition process in order to have an ad that is both aesthetically pleasing and effective in conveying the intended message.
For guidance and inspiration, consider the following resources:
Image Analysis in Machine Learning: [Guide to Image Analysis in Machine Learning]
GPT-4o API: Vision Use Cases: [Getting Started with OpenAI's API]
Learn how to use vision capabilities to understand images: [OpenAI vision]
YOLO object detection: [YOLO: Algorithm for Object Detection Explained]

A comprehensive overview of AI image generation technologies: [AltexSoft on AI Image Generation]
Detailed documentation and examples for using Automatic1111, a popular tool for image generation: [JarvisLabs on Automatic1111]
GitHub repository for Stable Diffusion Web UI: [AUTOMATIC1111/stable-diffusion-webui]
GitHub repository for ComfyUI, along with a guide on its usage: [ComfyUI by comfyanonymous]
A comprehensive guide for the ComfyUI user interface: [AndyHTu on ComfyUI]
Using the API with ComfyUI: [Medium Article by yushantripleseven]
GitHub repository for Fooocus, an open-source image generation tool, and a guide on its usage: [Fooocus by illyasviel]
Introduction to Fooocus: [Medium Article by genebernardin]
GitHub repository for Fooocus-API: [Fooocus-API by konieshadow]
Task 3:  Image Composition Agent
Adludio aims to explore innovative methods to organise and compose assets into advertisement frames that are not only aesthetically pleasing but also effectively convey the intended message. This task challenges participants to apply creative design principles and compositional strategies to assemble the previously generated images and text into coherent ad frames.
Agents and Their Tasks
Layout Planning Skill:
Asset Positioning: Determine the optimal positions for each asset within a frame.
Size Scaling: Adjust the size of each asset to ensure visual balance and alignment with the ad’s narrative.
Orientation Adjustment: (Optional) Consider the orientation of assets to enhance user engagement and perception.
Design Aesthetics Skill:
Color Coordination: Ensure that the colors of all assets are visually harmonious.
Consistency Check: Maintain consistency in design elements across different frames.
Visual Appeal Optimization: Enhance the overall visual appeal of the frame by applying design principles.
Narrative Flow Skill:
Sequence Arrangement: Arrange the assets in a logical sequence that conveys the ad’s narrative effectively.
Interaction Pathways: Develop pathways for user interactions within the ad, ensuring a smooth and engaging experience.
Multi-Path Management: (Optional) Handle multiple interaction paths for ads with branching narratives.

For inspiration and guidance on image composition, refer to the following resource:
A curated list of resources and tools for image composition: [Awesome Image Composition GitHub Repository]
3 Steps to an Effective Ad Layout: [Baer Performance Marketing]
Elements and Principles of Ad Design: [Rocketium Academy]
Elements and Principles of Design: [Creatopy Blog]

Task 4:  Building the Storyboard
For this final task, Adludio seeks innovative solutions for representing and visualizing the storyboard in the most aesthetically pleasing and informative manner possible. The challenge lies in effectively conveying the user flow within the ad, utilizing the frames composed in the previous task. The emphasis is on the creative use of placement and directional elements, guided by the concepts data, to depict the journey through the advertisement.
This task focuses on synthesizing the individual frames into a single storyboard image that represents the user flow through the advertisement.
Presenting User-Flow: Arrange the generated frames in a sequence that effectively conveys the progression of the ad's narrative, ensuring a logical and engaging user experience.
(Optional) Address Multi-Path Concepts: For ads with branching narratives or multiple user interaction paths, develop a strategy to incorporate these variations into the storyboard in a clear and coherent manner.
The objective is to understand the methodologies participants employ to construct the storyboard, focusing on the rationale behind the chosen representation method. Adludio is interested in learning why the selected approach is deemed superior to other alternatives, particularly in terms of aesthetics and information conveyance.
A successfully completed task will result in a storyboard that not only visually compels but also clearly communicates the intended user flow and narrative. Your submission should detail the thought process behind the storyboard construction, including the selection of specific compositional and directional strategies over other alternatives.
For reference and inspiration, you are encouraged to review existing storyboards provided to clients by Adludio, analyzing their structure, aesthetic qualities, and how they effectively depict user flow.

Tutorials Schedule
Overview
Tutorials should help you make continuous progress on the project. You should ask questions in slack and tutorial times such that you have roughly progress as follows 

Monday: Understanding week 12 challenge and getting familiarised with computer vision (CV) algorithms. Able to connect with the AWS machine and run code on GPU (tensorflow, keras, pytorch etc.). Think about accuracy and other metrics
Tuesday:  have understanding on deep learning segmentation and feature extraction from images and run example models. Define metrics and loss functions clearly. Start writing your literature review and understanding
Wednesday: test some CV algorithms on given data and start planning how to train ML models to predict KPIs from DL features.  Submit your interim report.
Thursday: Deep experimentation with deep learning and machine learning algorithms. Record your understanding, experimentation as part of your report. 
Friday: improve performance by playing with hyperparameters and architectures. Record your progress and result as part of your report.
Saturday: Finalise, and focus on presenting your findings in a report and github structure.
In the following, the colour purple indicates morning sessions, and blue indicates afternoon sessions.
Monday
Understanding week 12 challenge

Introduction to week11 challenge (Yabebal)

Key Performance:
Getting familiar with Computer Vision Algorithms
Able to connect to AWS machine and run code on GPU
Ability to reuse previous knowledge
Sharing of reference resources on computer visions for optimization
Tuesday
Understanding the creative assets data storyboard building

How to use CV Models and Algorithms (Emitnan)
Wednesday
How creative assets data is assembled and what has been done so far at Adludio(Rehmet)
Deliverables
Interim Submission - Wednesday 8 pm UTC
Link to your code in GitHub
Share a report that outlines your understanding of image analysis, and feature extraction.  Prepare this in a format that a 3rd-year student at a university can understand the basic concepts and reproduce your work.
Feedback
You may not receive detailed comments on your interim submission but will receive a grade.
Final Submission - Saturday 8 pm UTC
Link to your code in GitHub that you use to solve tasks 2 & 3
A blog post entry (which you can submit for example to Medium publishing) in the form of a PDF report. 

Feedback
You will receive comments/feedback in addition to a grade.



References
Key concepts:
Image Segmentation: The Basics and 5 Key Techniques (datagen.tech)
Train YOLOv7 Segmentation on Custom Data 🤔 | by Muhammad Rizwan Munawar | Augmented Startups | Sep, 2022 | Medium
Object Detection State of the Art 2022 | by Pedro Azevedo | Medium

Key Blogs:
Image Segmentation Guide - Almost everything you need to know about how image segmentation works | Fritz AI

Key Papers:
Towards automatic animated story boarding 
Artificial Intelligence in Digital Marketing: Insights from a Comprehensive Review
Full article: A Survey on Deep Learning-based Architectures for Semantic Segmentation on 2D Images (tandfonline.com)

Critical and Relevant:
OpenAI GPT-3 Text Embeddings - Really a new state-of-the-art in dense text embeddings? | by Nils Reimers | Medium
Models: 
What is Image Segmentation? - Hugging Face
Image segmentation  |  TensorFlow Core
MLOps (MLFlow, DVC, CML, Dagger)
https://docs.dagger.io/1200/local-dev/
https://cml.dev/
Data Version Control · DVC
MLflow - A platform for the machine learning lifecycle | MLflow

Huggingface tools and platforms
Github for ML Models: Introducing Skops (huggingface.co)
Build & Share Delightful Machine Learning Apps: Gradio
README - a Hugging Face Space by Gradio-Blocks 
Gradio 3.0 is Out! (huggingface.co)
Companies doing similar work
Luna | App Marketing Platform (is.com)
Creative Automation Production at Scale, Dynamic Ads Platform | Celtra
Best Creative Management Platforms in 2022: Compare Reviews on 60+ | G2

Unrelated but interesting
Advertisement Detection, Segmentation, and Classification for Newspaper Images and Website Snapshots

