const webcamElement = document.getElementById('webcam');
const canvasGrid = document.getElementById('grid');
const layerSelect = document.getElementById('layer');
const labelsDiv = document.getElementById('labels');
const modelRoot = "http://localhost:8080/Butterflies/";
const probability_threshold = 0.95;

async function app() {
    var preset = "";
    // Load the 'compact' model that was saved using Cognitive Services Custom Vision
    console.log('Loading model..');
    var model = await tf.loadGraphModel(modelRoot + "model.json");
    var response = await fetch(modelRoot + "labels.txt");
    var labels = await response.text();
    labels = labels.split("\n");
    labelsDiv.innerText = labels.toString();
    console.log('Successfully loaded model');

    // Capture image from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);

    // use classifier to identify what webcam sees in real time
    while (true) {
        var imgTensor = await webcam.capture();
        imgTensor = imgTensor.reshape([-1,224,224,3]).toFloat();
        var result = await model.predict(imgTensor);
        document.getElementById('prediction').innerText = `probability: ${result.toString()}`;
        var probabilities = result.dataSync();
        document.getElementById('predicted-name').innerText = "";
        document.getElementById('predicted-image').src = "Neither.png";
        for(var i=0; i < probabilities.length; i++) {
            if(probabilities[i] > probability_threshold) {
                document.getElementById('predicted-name').innerText = labels[i];
                document.getElementById('predicted-image').src = labels[i].replace(" ", "") + ".jpg";
            }
        }
        imgTensor.dispose();
        await sleep(500);
        await tf.nextFrame();
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

app();