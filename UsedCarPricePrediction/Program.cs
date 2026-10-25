using Microsoft.ML;
namespace UsedCarPricePrediction
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            string path = "used_cars_1000.csv";

            // 1. Load the data
            IDataView dataView = mlContext.Data.LoadFromTextFile<CarData>(path, hasHeader: true, separatorChar: ',');

            // 2. Split the data into training and testing sets
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // 3. Define the data processing pipeline

            var pipeLine = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: nameof(CarData.Make))
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: nameof(CarData.Model)))
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TransmissionEncoded", inputColumnName: nameof(CarData.Transmission)))
                           .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "FuelTypeEncoded", inputColumnName: nameof(CarData.FuelType)))
                           .Append(mlContext.Transforms.Concatenate("Features", "MakeEncoded", "ModelEncoded", nameof(CarData.Year), nameof(CarData.Mileage), nameof(CarData.EngineSize), nameof(CarData.Horsepower), "TransmissionEncoded", "FuelTypeEncoded", nameof(CarData.Doors)))
                           .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                           .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(CarData.Price), featureColumnName: "Features"));


            // 4. Train the model
            var model = pipeLine.Fit(split.TrainSet);

            // 5. Evaluate the model
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(CarData.Price));

            Console.WriteLine($"Model Evaluation Metrics:");
            Console.WriteLine($"R^2: {metrics.RSquared:0.###}");
            Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:0.###}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.###}");
            Console.WriteLine($"#########################################");

            // 6. Make a prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<CarData, CarPrediction>(model);

            //7. Sample data for prediction

            var sampleCar = new CarData
            {
                Make = "Toyota",
                Model = "Camry",
                Year = 2018,
                Mileage = 30000,
                EngineSize = 2.5f,
                Horsepower = 203,
                Transmission = "Automatic",
                FuelType = "Gasoline",
                Doors = 4
            };

            Console.WriteLine($"Sample Car: \n" +
                              $"Make: {sampleCar.Make} \n" +
                              $"Model: {sampleCar.Model} \n" +
                              $"Year: {sampleCar.Year}\n" +
                              $"Mileage: {sampleCar.Mileage}\n" +
                              $"Engine Size: {sampleCar.EngineSize}\n" +
                              $"Horsepower: {sampleCar.Horsepower}\n" +
                              $"Transmission: {sampleCar.Transmission}\n" +
                              $"Fuel Type: {sampleCar.FuelType}\n" +
                              $"Doors: {sampleCar.Doors}");
            Console.WriteLine($"#########################################");
            CarPrediction prediction = predictionEngine.Predict(sampleCar);
            Console.WriteLine($"Predicted Price: {prediction.Score:C2}");
        }
    }
}

