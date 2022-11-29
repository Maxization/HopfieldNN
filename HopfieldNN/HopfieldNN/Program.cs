using HopfieldNN;

//var input = DataHelper.ReadBmpFiles("../../../../../cat_bmp", 512, 512);

var datasets = new (string name, int width, int height)[] 
{ 
    //("animals-14x9", 14, 9),
    ("large-25x25", 25, 25),
    //("large-25x25.plus", 25,25),
    //("large-25x50", 25, 50),
    //("letters-14x20", 14, 20),
    //("letters-abc-8x12", 8, 12),
    //("OCRA-12x30-cut", 12, 30),
    //("small-7x7", 7, 7)
};

#region Clear files from folders
var folders = new List<string>
{
    "../../../../../1.SH",
    "../../../../../2.AH",
    "../../../../../3.SO",
    "../../../../../4.AO"
};

foreach(var folder in folders)
{
    DirectoryInfo di = new DirectoryInfo(folder);

    foreach (FileInfo file in di.GetFiles())
    {
        file.Delete();
    }
}
#endregion

var maxOjaIt = 1000;
var seed = 42;
var saveBitmaps = true;
var lr = 1e-7;

foreach(var dataset in datasets)
{
    var data = DataHelper.ReaderCSV($"../../../../../projekt2/{dataset.name}.csv");

    Console.WriteLine($"{dataset.name}:");
    Console.WriteLine("Synch Hebb");
    var nn = new HopfieldNetwork(LearningRule.Hebb, dataset.width, dataset.height, lr, maxOjaIt, seed, saveBitmaps);
    nn.savePath = "1.SH/out";
    nn.Train(data);
    var output = ProcessTest(nn, data, true);
    DisplayInfo(output.acc, output.stableCount, data[0].Length, data.Length);

    Console.WriteLine("\nAsync Hebb");
    nn = new HopfieldNetwork(LearningRule.Hebb, dataset.width, dataset.height, lr, maxOjaIt, seed, saveBitmaps);
    nn.savePath = "2.AH/out";
    nn.Train(data);
    output = ProcessTest(nn, data, false);
    DisplayInfo(output.acc, output.stableCount, data[0].Length, data.Length);

    Console.WriteLine("\nSynch Oja");
    nn = new HopfieldNetwork(LearningRule.Oja, dataset.width, dataset.height, lr, maxOjaIt, seed, saveBitmaps);
    nn.Train(data);
    nn.savePath = "3.SO/out";
    output = ProcessTest(nn, data, true);
    DisplayInfo(output.acc, output.stableCount, data[0].Length, data.Length);

    Console.WriteLine("\nAsync Oja");
    nn = new HopfieldNetwork(LearningRule.Oja, dataset.width, dataset.height, lr, maxOjaIt, seed, saveBitmaps);
    nn.Train(data);
    nn.savePath = "4.AO/out";
    output = ProcessTest(nn, data, false);
    DisplayInfo(output.acc, output.stableCount, data[0].Length, data.Length);

    Console.WriteLine("\n\n\n");
}

void DisplayInfo(List<double> accs, int stableCount, int neuronCount, int dataSize)
{
    Console.WriteLine($"Neuron Count: {neuronCount}");
    Console.WriteLine($"Dataset Size: {dataSize}");
    Console.WriteLine($"Stable states: {stableCount}");
    Console.WriteLine($"Min: {accs.Min()}");
    Console.WriteLine($"Max: {accs.Max()}");
    Console.WriteLine($"Avg: {accs.Sum() / accs.Count}");
}

(List<double> acc, int stableCount) ProcessTest(HopfieldNetwork nn, int[][] data, bool synch)
{
    var stableOutputs = new List<int[]>();
    var accs = new List<double>();
    foreach (var input in data)
    {
        var output = nn.Predict(input, synch);
        accs.Add(DataHelper.Accuracy(input, output));
        var newStable = true;
        foreach (var stable in stableOutputs)
        {
            if (Enumerable.SequenceEqual(output, stable))
            {
                newStable = false;
                break;
            }
        }

        if (newStable)
        {
            stableOutputs.Add(output);
        }
    }

    return (accs, stableOutputs.Count);
}
