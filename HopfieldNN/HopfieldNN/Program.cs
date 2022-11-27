// See https://aka.ms/new-console-template for more information


using HopfieldNN;

//var input = DataHelper.ReadBmpFiles("../../../../../cat_bmp", 512, 512);
int width = 14, height = 20;
var rng = new Random(42);
var nn = new HopfieldNetwork(width * height, "oja");
var data = DataHelper.ReaderCSV("../../../../../projekt2/letters-14x20.csv");
nn.Train(data);

var test = data[15];
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




