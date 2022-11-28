using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Newtonsoft.Json;

namespace HopfieldNN
{
    public static class DataHelper
    {
        public static int[] RandomVector(int n)
        {
            var rng = new Random();
            var v = new int[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = rng.Next() % 2 * 2 - 1;
            }
            return v;
        }

        public static int[] Blur(int[] data)
        {
            var blurred = (int[]) data.Clone();
            var rng = new Random();
            for (int i = 0; i < data.Length/10; i++)
            {
                var ind = rng.Next(data.Length - 1);
                data[ind] = -data[ind];
            }

            return blurred;
        }
        public static float Accuracy(int[] exp, int[] act)
        {
            int correct = 0;
            for (int i = 0; i < exp.Length; i++)
            {
                if (exp[i] == act[i])
                {
                    correct++;
                }
            }
            return (float)correct / exp.Length * 100;
        }

        public static int[][] StableInputs(int n)
        {
            int[][] data = new int[][]{};
            int bits = 12;
            for (int j = 0; j < Math.Pow(2, bits); j++)
            {
                var d = new int[n] ;
                var chars = n / bits;
                BitArray b = new BitArray(new int[] { j });
                for (int k = 0; k < bits; k++)
                {
                    for (int l = k * chars; l < k * chars + chars; l++)
                    {
                        d[l] = b[k]?1:-1;
                    }
                }
                if (j%10000 == 0)
                    Console.WriteLine($"{j/Math.Pow(2, bits)}");
                d[n - 1] = d[n - 2];
                data = data.Append(d).ToArray();
            }

            return data;
        }
        public static void Print(int[] data, int width, int height)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (data[i * width + j] == 1)
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
        }
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
            }

            return data.ToArray();
        }

        public static HopfieldNetwork HopfieldNetworkFromFile(string name)
        {
            var dir = "../../../../../trained";
            var json = File.ReadAllText($"{dir}/{name}.json");
            var  obj = JsonConvert.DeserializeObject<HopfieldNetwork>(json);
            return obj;
        }
        
        //public static void CreateBitmap(int[] input, int height, int width, int ind, string name, string suffix = "")
        //{
        //    Bitmap bm = new Bitmap(width, height);
        //    FastBitmap fbm = new FastBitmap(bm, ImageLockMode.WriteOnly);

        //    for (int i = 0; i < height; i++)
        //    {
        //        for (int j = 0; j < width; j++)
        //        {
        //            var color = input[i * width + j] > 0 ? Color.Black : Color.White;
        //            fbm[j, i] = new Span<byte>(BitConverter.GetBytes(color.ToArgb()));
        //        }
        //    }
        //    fbm.Dispose();

        //    bm.Save($"../../../../../outputs/{name}{ind}{suffix}.bmp", ImageFormat.Bmp);
        //}

        //public static Bitmap BitmapTo1Bpp(Bitmap img)
        //{
        //    int w = img.Width;
        //    int h = img.Height;
        //    Bitmap bmp = new Bitmap(w, h, PixelFormat.Format1bppIndexed);
        //    BitmapData data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadWrite, PixelFormat.Format1bppIndexed);
        //    byte[] scan = new byte[(w + 7) / 8];
        //    for (int y = 0; y < h; y++)
        //    {
        //        for (int x = 0; x < w; x++)
        //        {
        //            if (x % 8 == 0) scan[x / 8] = 0;
        //            Color c = img.GetPixel(x, y);
        //            if (c.GetBrightness() >= 0.5) scan[x / 8] |= (byte)(0x80 >> (x % 8));
        //        }
        //        Marshal.Copy(scan, 0, (IntPtr)((long)data.Scan0 + data.Stride * y), scan.Length);
        //    }
        //    bmp.UnlockBits(data);
        //    return bmp;
        //}

        //public static int[][] ReadBmpFiles(string folderPath, int width, int height)
        //{
        //    string[] images = Directory.GetFiles(folderPath);
        //    var result = new int[images.Length][];

        //    for (int i = 0; i < images.Length; i++)
        //    {
        //        Bitmap bm = new Bitmap(images[i]);
        //        FastBitmap fbm = new FastBitmap(bm, ImageLockMode.ReadOnly);
        //        var image = new int[width * height];
        //        for (int q = 0; q < height; q++)
        //        {
        //            for (int p = 0; p < width; p++)
        //            {
        //                var color_byte = fbm[p, q];
        //                int value = color_byte[0] > 0 ? -1 : 1;
        //                image[q * width + p] = value;
        //            }
        //        }
        //        result[i] = image;
        //    }

        //    return result;
        //}
    }
}
