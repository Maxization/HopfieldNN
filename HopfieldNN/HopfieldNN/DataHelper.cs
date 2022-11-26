using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HopfieldNN
{
    public static class DataHelper
    {
        public static int[][] ReaderCSV(string file)
        {
            var parser = new Microsoft.VisualBasic.FileIO.TextFieldParser(file);
            parser.TextFieldType = Microsoft.VisualBasic.FileIO.FieldType.Delimited;
            parser.SetDelimiters(new string[] { "," });

            var data = new List<int[]>();
            while (!parser.EndOfData)
            {
                var row = parser.ReadFields();
                if (row != null)
                {
                    var values = Array.ConvertAll(row, x => int.Parse(x));
                    data.Add(values);
                }

                //TEST
                if (data.Count > 3) break; 
            }

            return data.ToArray();
        }
        public static void CreateBitmap(int[] input, int height, int width, int ind)
        {
            Bitmap bm = new Bitmap(width, height);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    var color = input[i * width + j] > 0 ? Color.Black : Color.White;
                    bm.SetPixel(j, i, color);
                }
            }

            bm.Save($"test{ind}.bmp", ImageFormat.Bmp);
        }
    }
}
