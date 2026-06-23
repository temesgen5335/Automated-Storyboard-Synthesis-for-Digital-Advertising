# 10academy CV Challenge — July 2024 (refined brief)

> Extracted verbatim from the provided .docx for durable context. This is the more modern 3-task framing the project follows.

Automated Storyboard Synthesis: A Comprehensive Machine Learning Framework for Text-to-Visual Transformation in Digital Advertising

Business objective  
Recent advancements in machine learning, natural language processing, and computer vision, alongside the development of Large Language Models (LLMs), have ushered in a new era of capabilities in the digital domain. These technologies enable the intricate processing and interpretation of data, facilitating the creation of detailed, dynamic content that bridges the gap between textual concepts and visual storytelling. The integration of these technologies not only simplifies the translation of complex ideas into tangible visuals but also enhances creativity and efficiency in content generation. The goal of this task is to harness these capabilities to transform textual descriptions of advertisement concepts and assets into detailed storyboards. This transformation process aims to visually depict the narrative flow and user interaction within advertisements, making the conceptualization of digital campaigns both more intuitive and impactful.

Adludio is at the forefront of online mobile advertising, specializing in the creation of interactive ads that resonate with viewers through dynamic content such as mini-games, videos, texts, and images. Adludio offers its clients a suite of services designed to maximize engagement and campaign performance:
Collection of comprehensive briefs detailing brand identity, advertising objectives, guidelines, KPIs, objectives, and budget.
Design of interactive advertisements, leveraging a rich media creative toolkit.
Distribution of these creatives to targeted audiences via sophisticated real-time bidding for impressions on the open market.
Optimization of the creative design and targeting process through advanced machine learning algorithms to ensure maximum impact.

In this transformative era of advertising and recognizing the potential for technology to streamline and enhance the ad creation process, Adludio is embarking on an ambitious initiative to automate the end-to-end process of advertising production. This automation aims to significantly expedite the ideation and execution phases, enabling clients to swiftly launch their campaigns with minimal expenditure of time and resources. A key component of this automation involves the generation of potential creative concepts based on the client's brief. By leveraging advanced machine learning algorithms, Adludio intends to present clients with viable creative options rapidly, thereby reducing the traditional turnaround time from over a week to mere days. Your task, as part of this transformative process, is to architect and develop a cutting-edge machine learning solution that automates the conversion of textual advertisement concepts, assets descriptions into visually compelling storyboards. This solution should intelligently interpret the provided concepts and assets, generate relevant visual and textual assets, compose these assets into individual ad frames, and ultimately synthesize a cohesive storyboard that encapsulates the essence of the proposed ad campaign.



Background & Context

Start by reading these two blogs for a general understanding on the context and background of this challenge:  (NEEDS TO BE CHANGED)
Dynamic Creative & Content Optimization - DCO Marketing & Advertising (claravine.com)
5 Examples That Show How Machine Learning is Changing Digital Advertising
How Machine Learning Is Shaping The Future Of Advertising (forbes.com)

You may find this part helpful in understanding some of the technical terms you may find in the datasets.
Creative - an advertisement (ad) that users encounter and interact with while navigating a website or utilizing a mobile application powered by ads.
Concept - envisioned appearance and structure of the advertisement, outlining the creative idea behind it.
AdFrame - a creative is composed of several segments or scenes, each of which is called an AdFrame. These frames collectively tell the story or convey the message of the advertisement.
AdFormat - specifies the dimensions (width and height) of the space where the advertisement will be displayed. Common formats include Full Screen (FS) with dimensions of 320x480 pixels and Mid-Page Unit (MPU) with dimensions of 300x250 pixels.

The below diagram shows the flow of an event after we won an impression & an ad is displayed to the publisher.  Our Ads have multiple interactive screens animated with user interaction. Depending on the type of creative, an ad may have one or more screens. See the diagram below to understand how the user interacts with our ads.
		Fig 1: Example of a user interacting with an ad ( Event flow). 161925209550

Here are some examples of our latest ads:
Indica, ITC, Detran

Data

You will receive access to the following datasets, accompanied by a comprehensive breakdown of their contents and organizational structure:
Archive Folder: This archive features an 'Assets' folder, which, as the name implies, contains the images used to construct the creatives. Within this folder, subfolders correspond to different creative projects, each containing various assets integral to the creative. Notably, two crucial images, labeled 'landing' and 'endframe,' serve as the initial and final frames of the advertisement, respectively. These images, along with all other assets required to assemble these key frames, are included within the respective subfolders. [LINK]
Sample Concepts with Assets and Size Descriptions. A JSON file that outlines a series of concepts. Each object within the file includes the following details:
Concept: The creative idea's name.
Implementation: A detailed, frame-by-frame breakdown, providing visual representations and explanations for each segment.
Explanation: An overarching description of the advertisement's concept and the intended user flow.
Asset-Suggestions: For each frame, a curated list of three recommended assets is provided, detailing the category and a brief description of each suggested element.
Storyboard Examples: To offer insights into the standard of deliverables Adludio presents to its clients, a selection of sample storyboards is provided. These examples illustrate various approaches to storyboard composition and design:
StoryBoard-1 	(NEED LINKS HERE)
StoryBoard-2
StoryBoard-3
StoryBoard-4
StoryBoard-5

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



Introduction and Objective:
In the evolving landscape of digital advertising, the capability to automatically transform textual descriptions of advertisements into visual storyboards represents a significant leap towards creativity and efficiency. This challenge aims to leverage the latest advancements in machine learning, image, and text generation technologies to automate the storyboard creation process. By providing you with textual inputs detailing the concept, assets, and size of an advertisement, this initiative seeks to explore the potential of AI in streamlining the design process, enhancing ad engagement, and optimizing campaign performance. The ultimate objective is to develop a machine learning framework that can seamlessly convert textual ad descriptions into detailed, visually compelling storyboards that accurately reflect the intended user flow and narrative of an advertising campaign.

Task Breakdown:
To systematically address the challenge, we have divided the problem into a set of 3 consecutive tasks, each building upon the insights and outputs of the previous one. This structured approach is designed to guide you through the process of transforming textual inputs into a comprehensive storyboard that visually narrates the ad campaign's flow.

Task 1: Image and Text Generation
Adludio is seeking innovative solutions for generating images and text that appear as realistic and natural as possible. You are encouraged to explore and identify creative methods that can produce high-quality visual and textual content, aligning closely with the initial advertisement concepts provided. A deep understanding of image generation models, their operational mechanisms, and the utilization of available tools through automated processes, such as APIs, will be crucial in achieving the desired outcomes.
You will begin with the foundational task of generating visual and textual assets based on the provided descriptions. This task is crucial for creating the building blocks of the storyboard.
Image Generation:
Explore and implement methods for generating images that align with the given descriptions. Consider the aspect ratio and other properties that may affect the visual appeal and relevance of the images.
(Optional) Investigate techniques for refining image prompts to improve the quality and specificity of generated images.

Text Generation:
Develop strategies for converting provided text into images while also considering size(width and height)
(Optional) Delve into the details of font and other visual properties to enhance readability and aesthetic alignment with the ad's theme.
The primary aim is to discover and implement effective mechanisms for generating assets that convincingly mimic real-life scenarios. Adludio is particularly interested in the methodologies selected for this purpose, the rationale behind these choices, and the inventive approaches employed to create assets that are both realistic and engaging. The ability to automate this process efficiently, ensuring a seamless integration into the storyboard creation workflow, will be a key factor.
Successfully completing this task means you have developed a strategy that effectively utilizes image and text conversion technologies/mechanisms to produce assets that are close or indistinguishable to real-life objects and advertising standards.
For guidance and inspiration, consider the following resources:
A comprehensive overview of AI image generation technologies: [AltexSoft on AI Image Generation]
Detailed documentation and examples for using Automatic1111, a popular tool for image generation: [JarvisLabs on Automatic1111]
GitHub repository for Stable Diffusion Web UI: [AUTOMATIC1111/stable-diffusion-webui]
GitHub repository for ComfyUI, along with a guide on its usage: [ComfyUI by comfyanonymous]
A comprehensive guide for the ComfyUI user interface: [AndyHTu on ComfyUI]
Using the API with ComfyUI: [Medium Article by yushantripleseven]
GitHub repository for Fooocus, an open-source image generation tool, and a guide on its usage: [Fooocus by lllyasviel]
Introduction to Fooocus: [Medium Article by genebernardin]
GitHub repository for Fooocus-API: [Fooocus-API by konieshadow]
Task 2: Image Composition
Adludio aims to explore innovative methods to organize and compose generated images into advertisement frames that are not only aesthetically pleasing but also effectively convey the intended message. This task challenges participants to apply creative design principles and compositional strategies to assemble the previously generated images and text into coherent ad frames.
With the assets generated, the next step involves constructing each frame of the advertisement by determining the optimal placement and size of each asset.
Identifying Location and Size: Develop a method to dynamically position and size each asset within a frame, ensuring visual harmony and adherence to the ad's narrative flow.
(Optional) Consider additional factors such as orientation, which may impact the user's engagement and perception of the ad.
The focus is on uncovering unique approaches to composition and organization that enhance the visual and communicative impact of advertisement frames. Participants should explain their chosen methodologies for composition, the rationale behind these choices, and how they evaluate the effectiveness and appeal of their final compositions.
Successfully completing this task involves demonstrating a thoughtful approach to composing Adframes, with a clear explanation of the compositional choices made and the reasoning behind them. Your compositions should be both visually appealing and aligned with the advertisement's objectives, showcasing your ability to integrate design principles into practical applications.
For inspiration and guidance on image composition, refer to the following resource:
A curated list of resources and tools for image composition: [Awesome Image Composition GitHub Repository]
3 Steps to an Effective Ad Layout: [Baer Performance Marketing]
Elements and Principles of Ad Design: [Rocketium Academy]
Elements and Principles of Design: [Creatopy Blog]

Task 3: Building the Storyboard
For this final task, Adludio seeks innovative solutions for representing and visualizing the storyboard in the most aesthetically pleasing and informative manner possible. The challenge lies in effectively conveying the user flow within the ad, utilizing the frames composed in the previous task. The emphasis is on the creative use of placement and directional elements, guided by the concepts data, to depict the journey through the advertisement.
This task focuses on synthesizing the individual frames into a single storyboard image that represents the user flow through the advertisement.
Presenting User-Flow: Arrange the generated frames in a sequence that effectively conveys the progression of the ad's narrative, ensuring a logical and engaging user experience.
(Optional) Address Multi-Path Concepts: For ads with branching narratives or multiple user interaction paths, develop a strategy to incorporate these variations into the storyboard in a clear and coherent manner.
The objective is to understand the methodologies participants employ to construct the storyboard, focusing on the rationale behind the chosen representation method. Adludio is interested in learning why the selected approach is deemed superior to other alternatives, particularly in terms of aesthetics and information conveyance.
A successfully completed task will result in a storyboard that not only visually compels but also clearly communicates the intended user flow and narrative. Your submission should detail the thought process behind the storyboard construction, including the selection of specific compositional and directional strategies over other alternatives.
For reference and inspiration, you are encouraged to review existing storyboards provided to clients by Adludio, analyzing their structure, aesthetic qualities, and how they effectively depict user flow.


