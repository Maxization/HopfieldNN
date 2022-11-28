using System.Collections;
using HopfieldNN;

//var input = DataHelper.ReadBmpFiles("../../../../../cat_bmp", 512, 512);
int width = 7, height = 7;
var dataset = $"small-{width}x{height}";
var mode = "hebb";
var networkName = $"{dataset}-{mode}";
var rng = new Random(42);

var data = DataHelper.ReaderCSV($"../../../../../projekt2/{dataset}.csv");
var nn = new HopfieldNetwork(width * height, mode);

    

nn.Train(data);
nn.Save(networkName);
//var nn =DataHelper.HopfieldNetworkFromFile($"{dataset}-{mode}");
float[] accs = new float[]{};
for (int j = 0; j < data.Length; j++)
{
    var test = (int[])data[j].Clone();
//DataHelper.CreateBitmap(test, 9, 14, 0);
    // for (int i = 0; i < width*height*0.25; i++)
    // {
    //     var ind = rng.Next(width * height - 1);
    //     //var value = rng.Next(1) < 1 ? -1 : 1;
    //     test[ind] = -test[ind];
    // }

//DataHelper.CreateBitmap(test, 9, 14, 1);

    var test2 = nn.Predict(test);

    //DataHelper.Print(data[j], width, height);
    var acc = DataHelper.Accuracy(data[j], test2);
    accs = accs.Append(acc).ToArray();
    Console.WriteLine($"{j} {acc:N2}");
}
Console.WriteLine($"{dataset}\t{width*height}\t{data.Length}\t{accs.Min():N2}\t{accs.Max():N2}\t{accs.Sum()/data.Length:N2}");
//DataHelper.CreateBitmap(test2, 9, 14, 2);




