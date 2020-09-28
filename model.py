import os
import json
import numpy as np
import boto3
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow.serving import Model
from dotenv import load_dotenv
load_dotenv()


def train_and_deploy(deploy_name, embedding_dim=256, units=1024, batch_size=32, epochs=20):
    """
    Start a Sagemaker Training job passing the parameters to the TensorFlow model. Deploy the model to a SageMaker
     endpoint.
    :return: SageMaker TensorFlow model object.
    """
    # Setup the training job.
    model_artifacts_location = os.getenv("S3_SAGEMAKER_ARTIFACTS")
    tf_estimator = TensorFlow(entry_point='rnn.py',
                              role=os.getenv("SAGEMAKER_ROLE"),
                              train_instance_count=1,
                              train_instance_type='ml.p2.xlarge',
                              framework_version='2.3.0',
                              model_dir='/opt/ml/model',
                              output_path=model_artifacts_location,
                              py_version='py37',
                              script_mode=True,
                              hyperparameters={
                                  'embed-dim': embedding_dim,
                                  'rnn-units': units,
                                  'batch-size': batch_size,
                                  'epochs': epochs
                              })

    # Start the training job.
    tf_estimator.fit()

    # Create the SageMaker Model.
    tf_estimator.create_model()

    # Deploy to endpoint.
    deploy(tf_estimator.model_data, deploy_name)

    return tf_estimator


def transcript_processing():
    """
    Initial processing of the transcript file to generate character mappings that are written to S3.
    """
    s3 = boto3.client('s3')

    raw = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key='main_transcript.txt')
    text = raw["Body"].read().decode("utf-8")

    # The unique characters in the file
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    # Save these resources to S3.
    s3_resource = boto3.resource('s3')
    char2idx_obj = s3_resource.Object(os.getenv("S3_BUCKET"), 'char2idx.json')
    char2idx_obj.put(
        Body=(bytes(json.dumps(char2idx).encode('UTF-8')))
    )

    idx2char_obj = s3_resource.Object(os.getenv("S3_BUCKET"), 'idx2char.json')
    idx2char_obj.put(
        Body=(bytes(json.dumps(idx2char.tolist()).encode('UTF-8')))
    )


def deploy(model_data, endpoint_name):
    """
    Deploys a SageMaker endpoint for a trained model.
    :param model_data: S3 location of the model artifacts.
    :param endpoint_name: Name to assign to the SageMaker endpoint.
    """
    # Create the model endpoint.
    model = Model(
        model_data=model_data,
        role=os.getenv("SAGEMAKER_ROLE"),
        image=os.getenv("SAGEMAKER_IMAGE_URI"))

    model.deploy(1, 'ml.t2.medium', endpoint_name=endpoint_name)


def test_endpoint():
    """
    Test the endpoint. Creates a sample prompt and sends it to our SageMaker endpoint.
    :return:
    """
    # Load model artifacts.
    s3 = boto3.client('s3')
    char2idx = json.loads(s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key="char2idx.json")["Body"].read().decode('utf-8'))
    idx2char = json.loads(s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key="idx2char.json")["Body"].read())

    prompt = "GEORGIA:\nStay sexy and"

    print(generate_text(prompt, char2idx, idx2char))


def softmax(x):
    """
    Converts SageMaker logits output to probabilities.
    :param x: Array of logit values.
    :return: Array of probabilities.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def generate_text(seed, char2idx, idx2char):
    """
    Uses seed text and invokes the SageMaker endpoint we created.
    :param seed:
    :param char2idx:
    :param idx2char:
    :return:
    """
    # Evaluation step (generating text using the learned model)
    runtime = boto3.client('runtime.sagemaker')

    # Number of characters to generate
    num_generate = 100

    # Converting our start string to numbers (vectorizing).
    seed_int = [char2idx[s] for s in seed]

    # Empty string to store our results
    output = []

    payload = ",".join([str(x) for x in seed_int])

    # Here batch size == 1
    new_char = ""
    while new_char not in ["!", ".", "?"]:
        response = runtime.invoke_endpoint(EndpointName='mfm-model-6', ContentType='text/csv', Body=payload)
        predictions = json.loads(response["Body"].read().decode("utf-8"))["predictions"][0]

        predictions = softmax(predictions[-1])

        experiments = np.random.multinomial(n=3, pvals=predictions)

        idx = np.argmax(experiments)
        new_char = idx2char[idx]

        if new_char in ["!", ".", "?"]:
            output.append(new_char)
            break
        elif len(output) >= num_generate:
            break
        else:
            output.append(new_char)

        payload += f",{str(idx)}"

    return seed + ''.join(output)
