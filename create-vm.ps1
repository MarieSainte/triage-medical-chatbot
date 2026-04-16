$vmNameBase = "medical-llm"
$imageFamily = "common-cu129-ubuntu-2204-nvidia-580"
$imageProject = "deeplearning-platform-release"

$zones = @(
    "us-central1-a",
    "us-central1-b",
    "us-central1-f",
    "us-east1-b",
    "europe-west4-a",
    "asia-southeast1-a"
)

$gpus = @(
    "nvidia-tesla-p100",
    "nvidia-tesla-v100"
)

foreach ($zone in $zones) {

    foreach ($gpu in $gpus) {

        $vmName = "$vmNameBase-$($zone)-$($gpu.Split('-')[-1])"

        Write-Host "`nTrying $vmName in $zone with $gpu ..." -ForegroundColor Cyan

        gcloud compute instances create $vmName `
            --zone=$zone `
            --machine-type=n1-standard-4 `
            --accelerator="type=$gpu,count=1" `
            --image-family=$imageFamily `
            --image-project=$imageProject `
            --boot-disk-size=50GB `
            --maintenance-policy=TERMINATE `
            --restart-on-failure

        if ($LastExitCode -eq 0) {
            Write-Host "`nSUCCESS: VM created -> $vmName" -ForegroundColor Green
            exit 0
        }
        else {
            Write-Host "FAILED in $zone with $gpu (Exit Code: $LastExitCode)" -ForegroundColor Yellow
        }
    }
}

Write-Host "`nNo GPU available in tested zones." -ForegroundColor Red