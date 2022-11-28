using HopfieldNN;
using System.Collections;
using System.Data;
using System.Drawing;

int width = 150, height = 200;
var dataset = $"large-{width}x{height}";
var mode = "oja";///oja

//var networkName = $"{dataset}-{mode}";
var rng = new Random();

var data = DataHelper.ReaderCSV($"../../../../../projekt2/{dataset}.csv");
//var data = DataHelper.ReadBmpFiles("../../../../../cat_bmp", width, height);

var nn = new HopfieldNetwork(mode, width, height);
nn.Train(data);
//nn.Save(networkName);
//var nn = DataHelper.HopfieldNetworkFromFile($"{dataset}-{mode}");

var it = 0;
foreach (var input in data)
{
    var corruptedInput = DataHelper.RandomInput(width * height, rng); //input;// DataHelper.CorruptData(0.5, input, rng);
    DataHelper.CreateBitmap(corruptedInput, height, width, it, "in");
    var output = nn.Predict(corruptedInput);
    DataHelper.CreateBitmap(output, height, width, it++, "out", "_");
}
