// See https://aka.ms/new-console-template for more information


using HopfieldNN;

//var input = DataHelper.ReadBmpFiles("../../../../../cat_bmp", 512, 512);

var nn = new HopfieldNetwork(9 * 14);
var data = DataHelper.ReaderCSV("../../../../../projekt2/animals-14x9.csv");
nn.Train(data);

var test = data[0];
DataHelper.CreateBitmap(test, 9, 14, 0);
var rng = new Random();
for (int i = 0; i < 30; i++)
{
    var ind = rng.Next(9 * 14 - 1);
    var value = rng.Next(1) < 1 ? -1 : 1;
    test[ind] = value;
}

DataHelper.CreateBitmap(test, 9, 14, 1);

var test2 = nn.Predict(test);

DataHelper.CreateBitmap(test2, 9, 14, 2);




