const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

async function app() {
    console.log('Loading mobilenet..');
    var model = await mobilenet.load();
    console.log('Successfully loaded model');

    // Capture image from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);
    // Train classifier - Reads an image from the webcam and associates it with a specific class index.
    const addExample = async classId => {
        // Capture an image from the web camera.
        const img = await webcam.capture();

        // Get intermediate activation of MobileNet 'conv_preds' and pass to KNN classifier.
        const activation = model.infer(img, 'conv_preds');
        classifier.addExample(activation, classId);

        // Dispose the tensor to release the memory.
        img.dispose();
    }

    // When clicking a button, add an example for that class.
    document.getElementById('thumbs-up').addEventListener('click', () => addExample(0));
    document.getElementById('thumbs-down').addEventListener('click', () => addExample(1));
    document.getElementById('neither').addEventListener('click', () => addExample(2));

    while (true) {
        if (classifier.getNumClasses() > 0) {
            const imgTensor = await webcam.capture();

            // Get the activation from mobilenet from the webcam image.
            const activation = model.infer(imgTensor, 'conv_preds');
            // Get the most likely class and confidence from the classifier module.
            const result = await classifier.predictClass(activation);

            const classes = ['Thumbs_up.png', 'Thumbs_down.png', 'Neither.png'];

            if (result.confidences[result.label] > 0.99)
                document.getElementById('prediction').src = classes[result.label];
            else
                document.getElementById('prediction').src = "Neither.png";

            document.getElementById('probability').innerText = `
                Probability: ${result.confidences[result.label]} (${classes[result.label]})`;

            // Dispose the tensor to release memory.
            imgTensor.dispose();
        }

        await tf.nextFrame();
    }
}

app();