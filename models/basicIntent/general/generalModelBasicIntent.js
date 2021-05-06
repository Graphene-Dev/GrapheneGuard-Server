const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const use = require('@tensorflow-models/universal-sentence-encoder');


var training = require( './generalModelBasicIntentTraining.json');
var testing = require( './generalModelBasicIntentTesting.json');
/*
import comments from './comment-training.json';
import comment_testing from './comment-test.json';
 */

const encodeData = data => {
    const sentences = data.map(comment => comment.text.toLowerCase());
    const trainingData = use.load()
        .then(model => {
            return model.embed(sentences)
                .then(embeddings => {
                    return embeddings;
                });
        })
        .catch(err => console.error('Fit Error:', err));

    return trainingData
};

const outputData = tf.tensor2d(training.map(comment => [
    comment.intent === 'buy' ? 1 : 0,
    comment.intent === 'none' ? 1 : 0,
]));

// Output: [1,0] or [0,1]

const model = tf.sequential();

// Add layers to the model
model.add(tf.layers.dense({
    inputShape: [512],
    activation: 'sigmoid',
    units: 2,
}));

model.add(tf.layers.dense({
    inputShape: [2],
    activation: 'sigmoid',
    units: 2,
}));

model.add(tf.layers.dense({
    inputShape: [2],
    activation: 'sigmoid',
    units: 2,
}));

// Compile the model
model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(.06), // This is a standard compile config
});


function run() {
    Promise.all([
        encodeData(training),
        encodeData(testing)
    ])
        .then(data => {
            const {
                0: training_data,
                1: testing_data,
            } = data;

            model.fit(training_data, outputData, { epochs: 200 })
                .then(history => {
                    model.predict(testing_data).print();
                });
        })
        .catch(err => console.log('Prom Err:', err));
};

// Call function
run();