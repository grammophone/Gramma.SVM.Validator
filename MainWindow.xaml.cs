using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading;
using System.Diagnostics;

using Gramma.Optimization;
using Gramma.Kernels;
using Gramma.SVM;
using System.Threading.Tasks;
using Gramma.SVM.CoordinateDescent;

namespace Gramma.SVM.Validator
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		#region Command definitions

		public static RoutedUICommand TrainCommand =
			new RoutedUICommand("Search", "Search", typeof(MainWindow));

		#endregion

		#region Auxilliary types

		public enum KernelType
		{
			Linear,
			Gaussian
		}

		public enum ClassifierType
		{
			SerialCoordinateDescent,
			PartitioningCoordinateDescent
		}

		public class UserSettings
		{
			public string TrainingFilename { get; set; }
			public string ValidationFilename { get; set; }
			public KernelType KernelType { get; set; }
			public double C { get; set; }
			public double σ2 { get; set; }
			public bool UseShrinking { get; set; }
			public ClassifierType ClassifierType { get; set; }
			public int ProcessorsCount { get; set; }

			public UserSettings()
			{
				this.TrainingFilename = String.Empty;
				this.ValidationFilename = String.Empty;
				this.C = 1.0;
				this.σ2 = 4.0;
				this.UseShrinking = true;
				this.ClassifierType = MainWindow.ClassifierType.SerialCoordinateDescent;
				this.ProcessorsCount = Environment.ProcessorCount;
			}
		}

		public class TextBoxTraceListener : TraceListener
		{
			private TextBox textBox;

			public TextBoxTraceListener(TextBox textBox)
			{
				if (textBox == null) throw new ArgumentNullException("textBox");

				this.textBox = textBox;
			}

			public override void Write(string message)
			{
				textBox.Dispatcher.Invoke((Action)delegate
				{
					textBox.AppendText(message);
					textBox.ScrollToEnd();
				});
			}

			public override void WriteLine(string message)
			{
				textBox.Dispatcher.Invoke((Action)delegate
				{
					textBox.AppendText(message + "\n");
					textBox.ScrollToEnd();
				});
			}
		}


		#endregion

		#region Private fields

		private bool isTraining;

		private static KernelType[] kernelTypeValues = (KernelType[])Enum.GetValues(typeof(KernelType));

		private static ClassifierType[] classifierTypeValues = (ClassifierType[])Enum.GetValues(typeof(ClassifierType));

		#endregion

		#region Construction

		public MainWindow()
		{
			this.Settings = new UserSettings();

			InitializeComponent();

			this.DataContext = this;

			Trace.Listeners.Add(new TextBoxTraceListener(traceTextBox));
		}

		#endregion

		#region Public properties

		public UserSettings Settings { get; private set; }

		public static KernelType[] KernelTypeValues
		{
			get
			{
				return kernelTypeValues;
			}
		}

		public static ClassifierType[] ClassifierTypeValues
		{
			get
			{
				return classifierTypeValues;
			}
		}

		#endregion

		#region Private methods

		private void browseTrainingFileButton_Click(object sender, RoutedEventArgs e)
		{
			Microsoft.Win32.OpenFileDialog dialog = new Microsoft.Win32.OpenFileDialog();
			dialog.DefaultExt = ".txt";
			dialog.Filter = "Text documents (.txt)|*.txt";

			Nullable<bool> result = dialog.ShowDialog(this);

			if (result == false) return;

			trainingFileTextBox.Text = dialog.FileName;
			trainingFileTextBox.GetBindingExpression(TextBox.TextProperty).UpdateSource();

			int dotPosition = dialog.FileName.IndexOf('.');

			string prefix;

			if (dotPosition > 0)
				prefix = dialog.FileName.Substring(0, dotPosition);
			else
				prefix = dialog.FileName;

			string validationFilename = prefix + ".test.txt";

			if (System.IO.File.Exists(validationFilename))
			{
				validationFileTextBox.Text = validationFilename;
				validationFileTextBox.GetBindingExpression(TextBox.TextProperty).UpdateSource();
			}
		}

		private void browseValidationFileButton_Click(object sender, RoutedEventArgs e)
		{
			Microsoft.Win32.OpenFileDialog dialog = new Microsoft.Win32.OpenFileDialog();
			dialog.DefaultExt = ".txt";
			dialog.Filter = "Text documents (.txt)|*.txt";

			Nullable<bool> result = dialog.ShowDialog(this);

			if (result == false) return;

			validationFileTextBox.Text = dialog.FileName;
		}

		private void CanExecuteTrain(object sender, CanExecuteRoutedEventArgs e)
		{
			e.CanExecute = !this.isTraining && BindingHelper.AreAllValidated(this);
		}

		private void ExecuteTrain(object sender, ExecutedRoutedEventArgs e)
		{
			if (!BindingHelper.AreAllValidated(this)) return;

			try
			{
				Task classificationTask = Task.Factory.StartNew(() =>
					{
						if (this.isTraining) return;

						this.isTraining = true;
						this.Dispatcher.Invoke((Action)delegate
						{
							CommandManager.InvalidateRequerySuggested();
							traceTextBox.Clear();
							validationScoreTextBox.Text = String.Empty;
							trainingScoreTextBox.Text = String.Empty;
						});

						var trainingPairs = DataSource.ReadSparseVectors(this.Settings.TrainingFilename);

						if (trainingPairs.Count == 0) return;

						Kernel<Gramma.Vectors.SparseVector> kernel;

						switch (this.Settings.KernelType)
						{
							case KernelType.Linear:
								kernel = new SparseLinearKernel();
								break;

							case KernelType.Gaussian:
								kernel = new SparseRbfKernel(this.Settings.σ2);
								break;

							default:
								return;
						}

						CoordinateDescentBinaryClassifier<Gramma.Vectors.SparseVector> classifier;

						switch (this.Settings.ClassifierType)
						{
							case ClassifierType.SerialCoordinateDescent:
								classifier = new SerialCoordinateDescentBinaryClassifier<Gramma.Vectors.SparseVector>(kernel);
								break;

							case ClassifierType.PartitioningCoordinateDescent:
								var partitioningClassifier = new PartitioningCoordinateDescentBinaryClassifier<Gramma.Vectors.SparseVector>(kernel);
								partitioningClassifier.MaxProcessorsCount = this.Settings.ProcessorsCount;
								classifier = partitioningClassifier;
								break;

							default:
								throw new ApplicationException("No classifier type is selected.");
						}

						//classifier =
							//new PartitioningCoordinateDescentBinaryClassifier<Gramma.Vectors.SparseVector>(
							//new SerialCoordinateDescentBinaryClassifier<Gramma.Vectors.SparseVector>(
							//new SequentialBinaryClassifier<Gramma.Vectors.SparseVector>(
							//new PartitioningCoordinateDescentBinaryClassifier<Gramma.Vectors.SparseVector>(
							//new CgChunkingLineSearchBinaryClassifier<Gramma.Vectors.SparseVector>(
							//new CgChunkingNewtonBinaryClassifier<Gramma.Vectors.SparseVector>(
							//new CgLineSearchBinaryClassifier<Gramma.Optimization.Vector>(
							//new CgNewtonBinaryClassifier<Gramma.Optimization.Vector>(
							//kernel);

						//classifier.Options.UseShrinking = false;

						classifier.SolverOptions.GradientThreshold = 0.002f;
						classifier.SolverOptions.UseShrinking = this.Settings.UseShrinking;

						//classifier.Options.GradientThreshold = 1e-3;
						//classifier.Options.ShrinkingPeriod = 3000;

						//classifier.SolverOptions.BarrierInitialScale = 1;
						//classifier.SolverOptions.BarrierScaleFactor = 10;
						//classifier.SolverOptions.DualityGap = 1e-6;

						//classifier.ChunkingOptions.ConstraintThreshold = 1e-5;
						//classifier.Options.GradientThreshold = 0.001;

						//classifier.ChunkingOptions.ConstraintThreshold = 1e-2;
						//classifier.SolverOptions.DualityGap = 1e-4;

						//classifier.SolverOptions.KrylovIterationsCountLogOffsetStart = -0.15;
						//classifier.SolverOptions.KrylovIterationsCountLogOffsetEnd = 0.25;
						//classifier.SolverOptions.KrylovIterationsCountLogOffsetStep = 0.10;

						try
						{
							classifier.Train(trainingPairs, this.Settings.C);

							//Trace.WriteLine(String.Format("Memory after training: {0}.", GC.GetTotalMemory(true)));

							Trace.WriteLine("Scoring training set...");

							double trainingScore = GetScore(classifier, trainingPairs);

							this.trainingScoreTextBox.Dispatcher.Invoke((Action)delegate
							{
								this.trainingScoreTextBox.Text = trainingScore.ToString();
							});

							if (this.Settings.ValidationFilename.Trim() != String.Empty)
							{
								Trace.WriteLine("Reading validation set...");

								var validationPairs = DataSource.ReadSparseVectors(this.Settings.ValidationFilename);

								Trace.WriteLine("Scoring validation set...");

								double validationScore = GetScore(classifier, validationPairs);

								this.validationScoreTextBox.Dispatcher.Invoke((Action)delegate
								{
									this.validationScoreTextBox.Text = validationScore.ToString();
								});
							}

							Trace.WriteLine("Done.");
						}
						catch (Exception ex)
						{
							this.Dispatcher.Invoke((Action)delegate
							{
								MessageBox.Show(
									this,
									ex.Message,
									ex.Source + ": Boom!",
									MessageBoxButton.OK,
									MessageBoxImage.Stop);
							});
						}
						finally
						{
							this.isTraining = false;
							this.Dispatcher.Invoke((Action)delegate
							{
								CommandManager.InvalidateRequerySuggested();
							});
						}
					});

				classificationTask.ContinueWith((task) =>
				{
					Exception ex = task.Exception;

					if (ex != null)
					{
						if (ex is AggregateException)
						{
							var aggregateException = (AggregateException)ex;

							var messageBuilder = new StringBuilder();

							foreach (Exception innerException in aggregateException.InnerExceptions)
							{
								messageBuilder.Append(innerException.Message);

								if (String.IsNullOrEmpty(innerException.Source))
								{
									messageBuilder.Append(String.Format(" (Source: {0})", innerException.Source));
								}

								messageBuilder.AppendLine();
							}

							this.Dispatcher.Invoke((Action)delegate
							{
								MessageBox.Show(messageBuilder.ToString(), "Boom!", MessageBoxButton.OK, MessageBoxImage.Stop);
							});
						}
						else
						{
							this.Dispatcher.Invoke((Action)delegate
							{
								MessageBox.Show(
									this,
									ex.Message,
									ex.Source + ": Boom!",
									MessageBoxButton.OK,
									MessageBoxImage.Stop);
							});
						}

						this.isTraining = false;
						this.Dispatcher.Invoke((Action)delegate
						{
							CommandManager.InvalidateRequerySuggested();
						});

					}
				});

			}
			catch (Exception ex)
			{
				MessageBox.Show(this, ex.Message, ex.Source, MessageBoxButton.OK, MessageBoxImage.Stop);
			}
		}

		private static double GetScore(
      BinaryClassifier<Gramma.Vectors.SparseVector> classifier, 
      IList<BinaryClassifier<Gramma.Vectors.SparseVector>.TrainingPair> trainingPairs)
		{
			int correctPredictions = 0;

			//correctPredictions =
			//  trainingPairs.Sum(p => Math.Sign(classifier.Discriminate(p.Item)) == (int)p.Class ? 1 : 0);

			for (int i = 0; i < trainingPairs.Count; i++)
			{
				var validationPair = trainingPairs[i];

				var discriminate = classifier.Discriminate(validationPair.Item);

				if (Math.Sign(discriminate) == (int)validationPair.Class)
					correctPredictions++;
			}

			double validationScore = 100.0 * (double)correctPredictions / (double)trainingPairs.Count;

			return validationScore;
		}

		private void ExecuteClose(object sender, ExecutedRoutedEventArgs e)
		{
			this.Close();
		}

		private void kernelTypeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
		{
			this.kernelRadiusTextBox.IsEnabled =
				(this.Settings.KernelType == KernelType.Gaussian);
		}

		private void classifierTypeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
		{
			switch (this.Settings.ClassifierType)
			{
				case ClassifierType.PartitioningCoordinateDescent:
					this.processorsTextBox.IsEnabled = true;
					break;

				default:
					this.processorsTextBox.IsEnabled = false;
					break;
			}
		}

		#endregion

	}
}
