using HopfieldNN;

//var input = DataHelper.ReadBmpFiles("../../../../../cat_bmp", 512, 512);
int width = 7, height = 7;
var dataset = $"small-{width}x{height}";
var mode = "oja";
var networkName = $"{dataset}-{mode}";
var rng = new Random(42);

var data = DataHelper.ReaderCSV($"../../../../../projekt2/{dataset}.csv");
var nn = new HopfieldNetwork(width * height, mode);
nn.Train(data);
nn.Save(networkName);
//var nn =DataHelper.HopfieldNetworkFromFile($"{dataset}-{mode}");

var test = data[1];
//DataHelper.CreateBitmap(test, 9, 14, 0);
for (int i = 0; i < 30; i++)
{
    var ind = rng.Next(width * height - 1);
    var value = rng.Next(1) < 1 ? -1 : 1;
    test[ind] = value;
}

//DataHelper.CreateBitmap(test, 9, 14, 1);

var test2 = nn.Predict(test);
for (int i = 0; i < height; i++)
{
    for (int j = 0; j < width; j++)
    {
        if (test2[i * width + j] == 1)
        {
            Console.Write("O");
        }
        else
        {
            Console.Write(".");
        }
    }
    Console.WriteLine();
}
//DataHelper.CreateBitmap(test2, 9, 14, 2);




