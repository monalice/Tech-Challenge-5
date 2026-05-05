param(
    [ValidateSet("init", "validate", "plan", "apply", "all")]
    [string]$Action = "all",

    [string]$TerraformDir = "infra/terraform",

    [string]$VarFile = "terraform.tfvars",

    [switch]$AutoApprove
)

$ErrorActionPreference = "Stop"

function Assert-CommandExists {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Comando '$Name' não encontrado no PATH. Instale o Terraform e tente novamente."
    }
}

function Run-Step {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    Write-Host "`n==> $Label" -ForegroundColor Cyan
    & $Command

    if ($LASTEXITCODE -ne 0) {
        throw "Falha na etapa '$Label' (exit code $LASTEXITCODE)."
    }
}

Assert-CommandExists -Name "terraform"

$repoRoot = Get-Location
$tfPath = Join-Path $repoRoot $TerraformDir

if (-not (Test-Path $tfPath)) {
    throw "Diretório Terraform não encontrado: $tfPath"
}

Push-Location $tfPath

try {
    $cliConfigPath = Join-Path $tfPath ".terraformrc"
    $previousCliConfig = $env:TF_CLI_CONFIG_FILE

    if (Test-Path $cliConfigPath) {
        $env:TF_CLI_CONFIG_FILE = $cliConfigPath
        Write-Host "Usando TF_CLI_CONFIG_FILE: $cliConfigPath" -ForegroundColor DarkCyan
    }

    $varFilePath = Join-Path $tfPath $VarFile

    if (($Action -eq "plan" -or $Action -eq "apply" -or $Action -eq "all") -and -not (Test-Path $varFilePath)) {
        throw "Arquivo de variáveis não encontrado: $varFilePath"
    }

    if ($Action -eq "init" -or $Action -eq "all") {
        Run-Step -Label "terraform init" -Command { terraform init }
    }

    if ($Action -eq "validate" -or $Action -eq "all") {
        Run-Step -Label "terraform validate" -Command { terraform validate }
    }

    if ($Action -eq "plan" -or $Action -eq "all") {
        Run-Step -Label "terraform plan" -Command { terraform plan -var-file=$VarFile -out=tfplan }
    }

    if ($Action -eq "apply" -or $Action -eq "all") {
        if (-not (Test-Path (Join-Path $tfPath "tfplan"))) {
            Run-Step -Label "terraform plan (pré-apply)" -Command { terraform plan -var-file=$VarFile -out=tfplan }
        }

        if ($AutoApprove) {
            Run-Step -Label "terraform apply (auto-approve)" -Command { terraform apply -auto-approve tfplan }
        }
        else {
            Run-Step -Label "terraform apply" -Command { terraform apply tfplan }
        }
    }

    Write-Host "`nDeploy Terraform finalizado com sucesso." -ForegroundColor Green
}
finally {
    if ($null -ne $previousCliConfig -and $previousCliConfig -ne "") {
        $env:TF_CLI_CONFIG_FILE = $previousCliConfig
    }
    else {
        Remove-Item Env:\TF_CLI_CONFIG_FILE -ErrorAction SilentlyContinue
    }

    Pop-Location
}
