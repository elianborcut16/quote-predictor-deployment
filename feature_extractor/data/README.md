# Reference Data Files

This directory should contain the following reference data files:

1. `accounts.xlsx` - Company/account information with columns:
   - Accountname
   - Accountnumber
   - Address 1: city

2. `ships.xlsx` - Ship information with columns:
   - Name
   - (other ship details)

3. `bookings_2015.xlsx` - Historical bookings from 2015 with columns:
   - Organization
   - Date/Time
   - (other booking details)

## Format Requirements

These files should be Excel (.xlsx) format and contain the columns mentioned above.

If these files are not present, the feature extractor will still work but with reduced accuracy as it will fall back to default values and embedded reference data.

## Notes

- The original feature extraction logic from the Jupyter notebook has been converted to use these reference files
- When data files change, you don't need to update code - just replace these Excel files with updated versions
- The code will automatically detect and load the new data