# **Audio/Video Recording Transcription and Summarization**

## **Project Check Point**:

### **Abstract**:
Our project is a service that can produce summary, transcription and topical segmentation of provided video or audio. This line of services is always in demand due to large amounts of data people need to study in short amounts of time. Text is easier and faster to process and re-read than audio or video, which makes it more convenient in tasks of information retrieval. Our project utilizes Whisper to provide textual reprezentation of audio and video by demand.


#### 1. **Video/Audio Transcription**:
- We utilize OpenAI ASR Whisper to get the transcriptions of audio or video recordings
- Whisper was chosen over its alternatives, because it is trained on large dataset including diverse languages and examples
- It helps to widen the possible areas of usage


#### 2. **Text Segmentation**:
- Text segmentation is performed based on topics
- Whisper provides text broken down to sentences
- BERTopic then vectorizes and clusterizes these sentences
- Each sentence is then assigned to the most probable of topics detected by the model
- The result is a list of n lists, where n is a number of topics
- These lists, in turn, contain segments with timecodes, transcriptions and speakers

#### 3. **Summarization**
- Text summarization is performed by LLM
- Prompt engineering is used to increase the efficiency
