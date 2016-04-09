using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using Grammophone.SVM;
using Grammophone.Optimization;
using System.Globalization;
using Grammophone.Vectors;

namespace Grammophone.SVM.Validator
{
	public static class DataSource
	{
		public static IList<BinaryClassifier<Vector>.TrainingPair> ReadVectors(string filename)
		{
			using (FileStream stream = new FileStream(filename, FileMode.Open, FileAccess.Read))
			{
				return ReadVectors(stream);
			}
		}

		public static IList<BinaryClassifier<Vector>.TrainingPair> ReadVectors(Stream stream)
		{
			if (stream == null) throw new ArgumentNullException("stream");

			int dimensions;

			stream.Seek(0, SeekOrigin.Begin);

			using (var reader = new StreamReader(stream))
			{
				dimensions = GetDimensions(reader);

				stream.Seek(0, SeekOrigin.Begin);

				return ReadVectorElements(reader, dimensions);
			}

		}

    public static IList<BinaryClassifier<SparseVector>.TrainingPair> ReadSparseVectors(
      string filename)
    {
      using (FileStream stream = new FileStream(filename, FileMode.Open, FileAccess.Read))
      {
        return ReadSparseVectors(stream);
      }
    }

    public static IList<BinaryClassifier<SparseVector>.TrainingPair> ReadSparseVectors(
      Stream stream)
    {
      if (stream == null) throw new ArgumentNullException("stream");

      using (var reader = new StreamReader(stream))
      {
        return ReadSparseVectorElements(reader);
      }

    }

    private static int GetDimensions(TextReader reader)
		{
			int dimensions = 0;

			for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
			{
				line = line.Trim();

				if (line.Length == 0) continue;
				
				dimensions = Math.Max(dimensions, GetDimensions(line));
			}

			return dimensions;
		}

		private static int GetDimensions(string line)
		{
			var tokens = line.Split();

			if (tokens.Length <= 1) return 0;

			var lastToken = tokens.Last();

			return GetElement(lastToken).Index + 1;
		}

		private static SparseVector.Entry GetElement(string elementToken)
		{
			string[] parts = elementToken.Split(':');

			return new SparseVector.Entry(Int32.Parse(parts[0]) - 1, Double.Parse(parts[1], CultureInfo.InvariantCulture));
		}

		private static IList<BinaryClassifier<Vector>.TrainingPair> ReadVectorElements(TextReader reader, int dimensions)
		{
			var trainingPairs = new List<BinaryClassifier<Vector>.TrainingPair>();

			for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
			{
				line = line.Trim();

				if (line.Length == 0) continue;

				string[] elementTokens = line.Split();

				var trainingPair = new BinaryClassifier<Vector>.TrainingPair();

				trainingPair.Item = new Vector(dimensions);

				trainingPair.Class = 
					Double.Parse(elementTokens[0]) > 0.0 ? BinaryClass.Positive : BinaryClass.Negative;

				for (int i = 1; i < elementTokens.Length; i++)
				{
					var element = GetElement(elementTokens[i]);

					trainingPair.Item[element.Index] = element.Value;
				}

				trainingPairs.Add(trainingPair);
			}

			return trainingPairs;
		}

    private static IList<BinaryClassifier<SparseVector>.TrainingPair> ReadSparseVectorElements(
      TextReader reader)
    {
      var trainingPairs = new List<BinaryClassifier<SparseVector>.TrainingPair>();

      for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
      {
        line = line.Trim();

        if (line.Length == 0) continue;

        string[] elementTokens = line.Split();

        var trainingPair = new BinaryClassifier<SparseVector>.TrainingPair();

        trainingPair.Class =
          Double.Parse(elementTokens[0]) > 0.0 ? BinaryClass.Positive : BinaryClass.Negative;

        var entries = new List<SparseVector.Entry>();

        for (int i = 1; i < elementTokens.Length; i++)
        {
          var element = GetElement(elementTokens[i]);

          entries.Add(element);
        }

        trainingPair.Item = new SparseVector(entries);

        trainingPairs.Add(trainingPair);
      }

      return trainingPairs;
    }

  }
}
