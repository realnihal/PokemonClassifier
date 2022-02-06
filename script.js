const classes = [
  'Abra',
 'Aerodactyl',
 'Alakazam',
 'Alolan Sandslash',
 'Arbok',
 'Arcanine',
 'Articuno',
 'Beedrill',
 'Bellsprout',
 'Blastoise',
 'Bulbasaur',
 'Butterfree',
 'Caterpie',
 'Chansey',
 'Charizard',
 'Charmander',
 'Charmeleon',
 'Clefable',
 'Clefairy',
 'Cloyster',
 'Cubone',
 'Dewgong',
 'Diglett',
 'Ditto',
 'Dodrio',
 'Doduo',
 'Dragonair',
 'Dragonite',
 'Dratini',
 'Drowzee',
 'Dugtrio',
 'Eevee',
 'Ekans',
 'Electabuzz',
 'Electrode',
 'Exeggcute',
 'Exeggutor',
 'Farfetchd',
 'Fearow',
 'Flareon',
 'Gastly',
 'Gengar',
 'Geodude',
 'Gloom',
 'Golbat',
 'Goldeen',
 'Golduck',
 'Golem',
 'Graveler',
 'Grimer',
 'Growlithe',
 'Gyarados',
 'Haunter',
 'Hitmonchan',
 'Hitmonlee',
 'Horsea',
 'Hypno',
 'Ivysaur',
 'Jigglypuff',
 'Jolteon',
 'Jynx',
 'Kabuto',
 'Kabutops',
 'Kadabra',
 'Kakuna',
 'Kangaskhan',
 'Kingler',
 'Koffing',
 'Krabby',
 'Lapras',
 'Lickitung',
 'Machamp',
 'Machoke',
 'Machop',
 'Magikarp',
 'Magmar',
 'Magnemite',
 'Magneton',
 'Mankey',
 'Marowak',
 'Meowth',
 'Metapod',
 'Mew',
 'Mewtwo',
 'Moltres',
 'MrMime',
 'Muk',
 'Nidoking',
 'Nidoqueen',
 'Nidorina',
 'Nidorino',
 'Ninetales',
 'Oddish',
 'Omanyte',
 'Omastar',
 'Onix',
 'Paras',
 'Parasect',
 'Persian',
 'Pidgeot',
 'Pidgeotto',
 'Pidgey',
 'Pikachu',
 'Pinsir',
 'Poliwag',
 'Poliwhirl',
 'Poliwrath',
 'Ponyta',
 'Porygon',
 'Primeape',
 'Psyduck',
 'Raichu',
 'Rapidash',
 'Raticate',
 'Rattata',
 'Rhydon',
 'Rhyhorn',
 'Sandshrew',
 'Sandslash',
 'Scyther',
 'Seadra',
 'Seaking',
 'Seel',
 'Shellder',
 'Slowbro',
 'Slowpoke',
 'Snorlax',
 'Spearow',
 'Squirtle',
 'Starmie',
 'Staryu',
 'Tangela',
 'Tauros',
 'Tentacool',
 'Tentacruel',
 'Vaporeon',
 'Venomoth',
 'Venonat',
 'Venusaur',
 'Victreebel',
 'Vileplume',
 'Voltorb',
 'Vulpix',
 'Wartortle',
 'Weedle',
 'Weepinbell',
 'Weezing',
 'Wigglytuff',
 'Zapdos',
 'Zubat',
];

// Check to see if TF.js is available
const tfjs_status = document.getElementById("tfjs_status");

if (tfjs_status) {
  tfjs_status.innerText = "Loaded TensorFlow.js - version:" + tf.version.tfjs;
}

let model; // This is in global scope

const loadModel = async () => {
  try {
    const tfliteModel = await tflite.loadTFLiteModel(
      "model_01.tflite"
    );
    model = tfliteModel; // assigning it to the global scope model as tfliteModel can only be used within this scope
    // console.log(tfliteModel);

    //  Check if model loaded
    if (tfliteModel) {
      model_status.innerText = "Model loaded";
    }
  } catch (error) {
    console.log(error);
  }

  // // Prepare input tensors.
  // const img = tf.browser.fromPixels(document.querySelector('img'));
  // const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);

  // // Run inference and get output tensors.
  // let outputTensor = tfliteModel.predict(input);
  // console.log(outputTensor.dataSync());
};
loadModel();

// Function to classify image
function classifyImage(model, image) {
  // Preprocess image
  image = tf.image.resizeBilinear(image, [180, 180]); // image size needs to be same as model inputs
  image = tf.expandDims(image);
  console.log(image);
  // console.log(model);

  // console.log(tflite.getDTypeFromTFLiteType("uint8")); // Gives int32 as output thus we cast int32 in below line
  // console.log(tflite.getDTypeFromTFLiteType("uint8"));
  console.log("converting image to different datatype...");
  image = tf.cast(image, "int32"); // Model requires uint8
  console.log("model about to predict...");
  const output = model.predict(image);
  const output_values = tf.softmax(output.arraySync()[0]);
  console.log("Arg max:");
  // console.log(output);
  console.log(output_values.arraySync());
  console.log("Output:");
  console.log(output.arraySync());
  console.log(output.arraySync()[0]); // arraySync() Returns an array to use

  // Update HTML
  predicted_class.textContent = output_values.argMax().arraySync();
  predicted_prob.textContent = output_values.max().arraySync() * 100 + "%";
}

// Image uploading
const fileInput = document.getElementById("file-input");
const image = document.getElementById("image");
const uploadButton = document.getElementById("upload-button");

function getImage() {
  if (!fileInput.files[0]) throw new Error("Image not found");
  const file = fileInput.files[0];

  // Get the data url from the image
  const reader = new FileReader();

  // When reader is ready display image
  reader.onload = function (event) {
    // Get data URL
    const dataUrl = event.target.result;

    // Create image object
    const imageElement = new Image();
    imageElement.src = dataUrl;

    // When image object loaded
    imageElement.onload = function () {
      // Display image
      image.setAttribute("src", this.src);

      // Log image parameters
      const currImage = tf.browser.fromPixels(imageElement);

      // Classify image
      classifyImage(model, currImage);
    };

    document.body.classList.add("image-loaded");
  };

  // Get data url
  reader.readAsDataURL(file);
}

// Add listener to see if someone uploads an image
fileInput.addEventListener("change", getImage);
uploadButton.addEventListener("click", () => fileInput.click());

// console.log(tf.browser.fromPixels(fileInput.files[0]).print());

// console.log(tf.browser.fromPixels(document.querySelector("image")));

// const test_image = new ImageData(1, 1);
// test_image.data[0] = 100;
// test_image.data[1] = 150;
// test_image.data[2] = 200;
// test_image.data[3] = 255;

// tf.browser.fromPixels(test_image).print();
