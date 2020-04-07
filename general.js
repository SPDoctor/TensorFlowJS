const webcamElement = document.getElementById('webcam');

async function app() {
    console.log('Loading mobilenet..');
    var model = await mobilenet.load();
    console.log('Successfully loaded model');

    // Capture image from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);

    // use classifier to identify what webcam sees in real time
    while (true) {
        var imgTensor = await webcam.capture();
        var results = await model.classify(imgTensor);
        var prediction = results.map(result => result.className + " (" + result.probability + ")    \n");
        document.getElementById('prediction').innerText = prediction;
        // Dispose the tensor to release the memory.
        imgTensor.dispose();
        await tf.nextFrame();
    }

}

app();