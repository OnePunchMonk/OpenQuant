#!/usr/bin/env pwsh
# Security check script to verify .env is properly gitignored

Write-Host "`n🔒 Security Check - API Key Safety`n" -ForegroundColor Cyan

# Check 1: Does .env exist?
if (Test-Path .env) {
    Write-Host "✓ .env file found" -ForegroundColor Green
} else {
    Write-Host "⚠ .env file not found - you need to create it from .env.example" -ForegroundColor Yellow
    Write-Host "  Run: Copy-Item .env.example .env" -ForegroundColor Gray
}

# Check 2: Is .env in .gitignore?
$gitignoreContent = Get-Content .gitignore -Raw
if ($gitignoreContent -match '\.env') {
    Write-Host "✓ .env is listed in .gitignore" -ForegroundColor Green
} else {
    Write-Host "✗ WARNING: .env is NOT in .gitignore!" -ForegroundColor Red
    Write-Host "  Add '.env' to your .gitignore file immediately!" -ForegroundColor Red
}

# Check 3: Is .env actually ignored by git?
$gitCheckIgnore = git check-ignore .env 2>&1
if ($gitCheckIgnore -match '\.env') {
    Write-Host "✓ Git is ignoring .env file" -ForegroundColor Green
} else {
    Write-Host "⚠ Git may not be ignoring .env properly" -ForegroundColor Yellow
}

# Check 4: Is .env in git index?
$gitStatus = git ls-files .env 2>&1
if ($gitStatus -eq "") {
    Write-Host "✓ .env is NOT tracked by git" -ForegroundColor Green
} else {
    Write-Host "✗ DANGER: .env IS TRACKED by git!" -ForegroundColor Red
    Write-Host "  Run this immediately: git rm --cached .env" -ForegroundColor Red
}

# Check 5: Verify .env.example exists
if (Test-Path .env.example) {
    Write-Host "✓ .env.example template exists" -ForegroundColor Green
} else {
    Write-Host "⚠ .env.example not found" -ForegroundColor Yellow
}

Write-Host "`n📋 Summary:" -ForegroundColor Cyan
Write-Host "  • Always use .env for your real API keys" -ForegroundColor Gray
Write-Host "  • .env is gitignored and will NOT be pushed to GitHub" -ForegroundColor Gray
Write-Host "  • .env.example is the public template (no real keys)" -ForegroundColor Gray
Write-Host "  • Never commit API keys in code or config.yaml`n" -ForegroundColor Gray
