using System.Collections;
using HopfieldNN;

//var input = DataHelper.ReadBmpFiles("../../../../../cat_bmp", 512, 512);
int width = 7, height = 7;
var dataset = $"small-{width}x{height}";
var mode = LearningRule.Hebb;
var networkName = $"{dataset}-{mode}";
var rng = new Random(42);

var data = DataHelper.ReaderCSV($"../../../../../projekt2/{dataset}.csv");
var nn = new HopfieldNetwork(mode, width, height, 1e-7, 100, 42, true);

nn.Train(data);

foreach(var input in data)
{
    nn.Predict(input);
}
