const webcamElement = document.getElementById('webcam');
const canvasGrid = document.getElementById('grid');
const layerSelect = document.getElementById('layer');
const labelsDiv = document.getElementById('labels');
const modelRoot = "http://localhost:8080/Butterflies/";

async function app() {
    var preset = "";
    // Load the 'compact' model that was saved using Cognitive Services Custom Vision
    console.log('Loading model..');
    var model = await tf.loadGraphModel(modelRoot + "model.json");
    var response = await fetch(modelRoot + "labels.txt");
    labelsDiv.innerText = await response.text();
    console.log('Successfully loaded model');
    // set up layer picker
    for(var i=0; i < model.artifacts.modelTopology.node.length; i++) {
        var layer = model.artifacts.modelTopology.node[i];
        if(layer.op == "Conv2D" || layer.op == "Relu") {
            var option = document.createElement("option");
            option.text = layer.name;
            layerSelect.add(option);
        }
    }

    // Capture image from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);

    // use classifier to identify what webcam sees in real time
    while (true) {
        var imgTensor = await webcam.capture();

        imgTensor = imgTensor.reshape([-1,224,224,3]).toFloat();
        var result = await model.predict(imgTensor);
        document.getElementById('prediction').innerText = `probability: ${result.toString()}`;

        var layer = layerSelect.options[layerSelect.selectedIndex];
        if(layer.value != "none") {
            var grid = await model.execute(imgTensor, layer.value);
            grid = tf.transpose(grid,[1,2,3,0]);
            grid = gridify(grid);
            grid = tf.squeeze(grid);
            grid = tf.sigmoid(grid);
            await tf.browser.toPixels(grid, canvasGrid);
            grid.dispose();
            }
        imgTensor.dispose();
        await sleep(500);
        await tf.nextFrame();
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

// Convert a tensor of form `[H,W,C,CO]` to form `[1,H',W',CO] where the `C`
// rank is turned into a grid of objects to visualise activation layers of images.
function gridify(input, pad, I, J) {
    var pad = pad || 2;
    let x = tf.pad(input, [[pad,pad],[pad,pad],[0,0],[0,0]])
    let [H, W, C, CO] = input.shape;
    H = H + 2 * pad;
    W = W + 2 * pad;
    if (J === undefined) {
        J = Math.floor(Math.sqrt(C));
        while (C % J !== 0) J--;
        I = Math.floor(C / J);
    }
    if (I * J !== C) throw new Error("Can't make a grid.")
    x = tf.transpose(x, [2,0,1,3]);
    x = tf.reshape(x, [J,I * H, W, CO]);
    x = tf.transpose(x, [0,2,1,3])
    x = tf.reshape(x, [1, J * W, I * H, CO])
    x = tf.transpose(x,[0,2,1,3])
    return x  //: (1, H * I, W * J, CO)
}

app();