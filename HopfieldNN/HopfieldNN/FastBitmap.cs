using System.Drawing.Imaging;
using System.Drawing;

//https://stackoverflow.com/questions/2574564/save-a-32-bit-bitmap-as-1-bit-bmp-file-in-c-sharp
public unsafe class FastBitmap : IDisposable
{
    private Bitmap _bmp;
    private ImageLockMode _lockmode;
    private int _pixelLength;

    private Rectangle _rect;
    private BitmapData _data;
    private byte* _bufferPtr;

    public int Width { get => _bmp.Width; }
    public int Height { get => _bmp.Height; }
    public PixelFormat PixelFormat { get => _bmp.PixelFormat; }

    public FastBitmap(Bitmap bmp, ImageLockMode lockMode)
    {
        _bmp = bmp;
        _lockmode = lockMode;

        _pixelLength = Image.GetPixelFormatSize(bmp.PixelFormat) / 8;
        _rect = new Rectangle(0, 0, Width, Height);
        _data = bmp.LockBits(_rect, lockMode, PixelFormat);
        _bufferPtr = (byte*)_data.Scan0.ToPointer();
    }

    public Span<byte> this[int x, int y]
    {
        get
        {
            var pixel = _bufferPtr + y * _data.Stride + x * _pixelLength;
            return new Span<byte>(pixel, _pixelLength);
        }
        set
        {
            value.CopyTo(this[x, y]);
        }
    }

    public void Dispose()
    {
        _bmp.UnlockBits(_data);
    }
}