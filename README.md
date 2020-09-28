# My Favorite Murder Catchphrase Generator

What happens when you have a computer listen to dozens of My Favorite Murder episodes and then write its own MFM-inspired
catchphrases? That's what I tried to find out with this bot!

You can create your own catchphrase here:

[MFM Catchphrase Generator](https://mfm-stay-sexy-and.herokuapp.com/)

The model was trained with SageMaker using a custom TensorFlow model. A SageMaker endpoint is hosted to handle prediction
requests.

This repository contains the code for interacting with AWS Transcribe to create transcriptions of podcast episodes and
for the training / deployment of the SageMaker model to generate new catchphrases.

## Transcribing Podcast Audio Files with AWS Transcribe.

To generate data for the model to be trained against I used [AWS Transcribe](https://aws.amazon.com/transcribe/) to 
construct transcripts for dozens of My Favorite Murder Episodes.

The `transcriptions.py` file contains all the functions I used for interacting with Transcribe.

The two main functions are:
 
 - `transcribe_recording` loads the raw podcast recording to S3 and then starts the transcription job with AWS Transcribe.
 - `process_transcript` performs post-transcription processing on the Transcribe output to identify speaker labels and
 append output to the main transcript.

## Building a Text Generation Model with SageMaker.

For the neural network I largely followed the [TensorFlow RNN Text Generation Tutorial](https://www.tensorflow.org/tutorials/text/text_generation)
and adopted it to work with [AWS SageMaker](https://aws.amazon.com/sagemaker/). I deployed the custom TensorFlow model
using script mode within SageMaker. With script mode you define a script for defining and training your model along
with the image you want to use for the training. When you train your model SageMaker will setup a container environment
according to your specifications and train your model within that environment.

- `model.py` calls the SageMaker APIs to setup, train, and deploy the model endpoint. The `train_and_deploy` function
will call the training script to train the model. Then the model is deployed as a SageMaker endpoint that you can make
real-time predictions against.
- `rnn.py` is the script that is run to train the model with. This defines the RNN model, generates the input data, trains
the model and saves it.

### What I Learned.

This was my first time using Transcribe so I learned a few things along the way. First, while Transcribe can be configured
to identify multiple speakers, there is additional processing needed to map the speaker labels to the content of what
that speaker said. The `assign_speakers` function is responsible for handling this.

Next, the speaker labels provided by Transcribe are integers. There was the added task of identifying whether Karen or
Georgia is speaking. In the `karen_or_georgia` function I map the Transcribe speaker label to the correct Murderino. The
mapping is done by counting the number of times each speaker mentions Karen or Georgia. An assumption is made that a 
speaker will mention the other's name more often than their own. 

Finally, I recognized the Transcribe output wasn't perfect. I invested some time in trying to clean up a few areas where
I noticed there were some errors. Especially around speaker transitions there would be parts of the transcription that
overflowed into the next speaker's piece. Cleaning up these transcriptions is one area for future improvement to make
the service better.

Previously when I have used SageMaker it was through the Notebook interface. Utilizing the APIs to train / deploy my models
was a huge benefit for my workflow. 