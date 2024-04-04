using Grasshopper;
using Grasshopper.Kernel;
using System;
using System.Drawing;

namespace SolarGAN
{
    public class SolarGANInfo : GH_AssemblyInfo
    {
        public override string Name => "SolarGAN";

        //Return a 24x24 pixel bitmap to represent this GHA library.
        public override Bitmap Icon => null;

        //Return a short string describing the purpose of this GHA library.
        public override string Description => "";

        public override Guid Id => new Guid("cef8149a-7ea3-4d68-8769-bd3b3bdcf0bd");

        //Return a string identifying you or your company.
        public override string AuthorName => "";

        //Return a string representing your preferred contact details.
        public override string AuthorContact => "";
    }
}