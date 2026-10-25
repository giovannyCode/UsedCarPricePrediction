using Microsoft.ML.Data;

namespace UsedCarPricePrediction
{
    public class CarPrediction
    {
        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
