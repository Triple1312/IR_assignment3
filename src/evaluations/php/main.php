<?php

class DataFrame {
    public array $columnNames = [];
    public array $data = [];

    public function fromCSV(string $path) : void {
        $file = file_get_contents($path);
        $lines = explode("\n", $file);
        $this->columnNames = explode(",", $lines[0]);
        for ($i = 1; $i < count($lines); $i++) {
            array_push($this->data, explode(",", $lines[$i]));
        }
    }
}


class SystemEvaluator {

    public static function precision(array $relevant, array $predictions, int $k): float {
        $precision = 0;
        for ($i = 0; $i < $k; $i++) {
            if (in_array($predictions[$i], $relevant)) {
                $precision++;
            }
        }
        return $precision / $k;
    }

    public static function recall(array $relevant, array $predictions, int $k): float {
        $recall = 0;
        for ($i = 0; $i < $k; $i++) {
            if (in_array($predictions[$i], $relevant)) {
                $recall++;
            }
        }
        return $recall / count($relevant);
    }

    public static function mean_average_precision(array $relevant, array $predictions, int $k): float {
        $precision = 0;
        for ($i = 0; $i < count($relevant); $i++) {
            $precision += self::precision($relevant[$i], $predictions[$i], $k);
        }
        return $precision / count($relevant);

    }

    public static function mean_average_reacall(array $relevant, array $predictions, int $k): float {
        $recall = 0;
        for ($i = 0; $i < count($relevant); $i++) {
            $recall += self::recall($relevant[$i], $predictions[$i], $k);
        }
        return $recall / count($relevant);
    }

}


$results_df = new DataFrame();
$results_df->fromCSV("results.csv");

$predictions_df = new DataFrame();
$predictions_df->fromCSV("predictions.csv");

echo "loaded dataframes\n";

$results = [];
$predictions = [];

$current_id = $results_df->data[0][0];
$results_tmp = [];
for ($i = 0; $i < count($results_df->data); $i++) {
    if ($results_df->data[$i][0] != $current_id) {
        array_push($results, $results_tmp);
        $current_id = $results_df->data[$i][0];
        $results_tmp = [];
//        echo $results_df->data[$i][1] . "\n";
        array_push($results_tmp, $results_df->data[$i][1]);
    }
    else {
        array_push($results_tmp, $results_df->data[$i][1]);
    }
}

$current_id = $predictions_df->data[0][0];
$predictions_tmp = [];
for ($i = 0; $i < count($predictions_df->data); $i++) {
    if ($predictions_df->data[$i][0] != $current_id) {
        array_push($predictions, $predictions_tmp);
        $current_id = $predictions_df->data[$i][0];
        print_r($predictions_tmp);
        $predictions_tmp = [];
        array_push($predictions_tmp, $predictions_df->data[$i][1]);
    }
    else {
        array_push($predictions_tmp, $predictions_df->data[$i][1]);
    }
}

echo "precision@1: " . SystemEvaluator::mean_average_precision($results, $predictions, 1) . "\n";
echo "precision@3: " . SystemEvaluator::mean_average_precision($results, $predictions, 3) . "\n";
echo "precision@5: " . SystemEvaluator::mean_average_precision($results, $predictions, 5) . "\n";
echo "precision@10: " . SystemEvaluator::mean_average_precision($results, $predictions, 10) . "\n";

echo "recall@1: " . SystemEvaluator::mean_average_reacall($results, $predictions, 1) . "\n";
echo "recall@3: " . SystemEvaluator::mean_average_reacall($results, $predictions, 3) . "\n";
echo "recall@5: " . SystemEvaluator::mean_average_reacall($results, $predictions, 5) . "\n";
echo "recall@10: " . SystemEvaluator::mean_average_reacall($results, $predictions, 10) . "\n";






















