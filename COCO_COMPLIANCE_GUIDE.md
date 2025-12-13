# COCO Compliance Guide

This guide explains how to verify and use the COCO-compliant implementation.

## Changes Made

### 1. ✅ COCO Archive Data Loading (IMPLEMENTED)

The `load_coco_archive_data()` method now:
- Searches for `.dat` files in `coco_archive/` directory
- Parses COCO `.dat` file format
- Converts to result dictionary format
- Automatically copies `.dat` files to `coco_logs/` for cocopp processing

**File format expected:**
```
coco_archive/
  ├── CMA-ES_bbob_f001_i01_d02.dat
  ├── CMA-ES_bbob_f001_i02_d02.dat
  ├── CMA-ES-LQ_bbob_f001_i01_d02.dat
  └── ...
```

### 2. ✅ Observer API (FIXED with fallback)

The Observer API now tries multiple constructor formats:
1. `Observer(suite, folder_string)` - Standard 2-parameter
2. `Observer(suite, folder_string, algorithm_name)` - 3-parameter (if supported)
3. Fallback with algorithm name in folder structure

## How to Verify Observer API

### Step 1: Run the verification script

```bash
python verify_coco_observer.py
```

This script will:
- Test different Observer constructor formats
- Check if `.dat` files are generated correctly
- Show which API format works with your cocoex version

### Step 2: Check the output

The script will tell you:
- ✓ Which constructor format works
- ✓ If algorithm name setting works
- ✓ If `.dat` files are generated

### Step 3: Update code if needed

If the verification shows a different API format works, you may need to adjust line 332-348 in `run_cmaes_comparison.py`.

## How to Use COCO Archive Datasets

### Step 1: Download COCO Archive Data

1. Go to: https://github.com/numbbo/coco/tree/master/data-archive
2. Download `.dat` files for baseline algorithms:
   - CMA-ES
   - CMA-ES-LQ (LQ-CMA-ES)
   - CMA-ES-DTS (DTS-CMA-ES) - if available
   - CMA-ES-LMM (LMM-CMA-ES) - if available

### Step 2: Place Files in Archive Directory

Place downloaded `.dat` files in:
```
results/cmaes_comparison_TIMESTAMP/coco_archive/
```

### Step 3: Run Comparison

The code will automatically:
- Detect archive algorithms
- Load `.dat` files from `coco_archive/`
- Copy them to `coco_logs/` for cocopp processing
- Process all algorithms together with cocopp

## COCO Compliance Checklist

- [x] Using cocoex with observers
- [x] Observer attached before optimization
- [x] Observer finalized after optimization
- [x] Results saved in `.dat` format (via observers)
- [x] Using COCO archive datasets for baseline algorithms
- [x] Running new experiments only for novel algorithms
- [x] Using cocopp for post-processing
- [x] No custom plotting (removed)
- [x] JSON files optional (not COCO standard)

## Troubleshooting

### Observer API Issues

If you get errors about Observer constructor:
1. Run `python verify_coco_observer.py`
2. Check which format works
3. Update `run_single_test()` method accordingly

### Archive Data Not Found

If archive data is not loading:
1. Check files are in `coco_archive/` directory
2. Verify file naming: `Algorithm_bbob_f001_i01_d02.dat`
3. Check verbose output for exact pattern being searched

### cocopp Not Working

If cocopp fails:
1. Ensure `.dat` files exist in `coco_logs/`
2. Check cocopp is installed: `pip install cocopp`
3. Try running manually: `python -m cocopp coco_logs`

## Example Usage

```bash
# Run with AFN (novel) and CMA-ES (archive)
python run_cmaes_comparison.py --algorithms AFN,CMA-ES --functions 1,2,3 --dimensions 2,5 --verbose

# Verify Observer API first
python verify_coco_observer.py
```

