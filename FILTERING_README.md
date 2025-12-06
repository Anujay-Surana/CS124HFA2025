# Demographic Filtering System - Quick Reference

## What It Does

Filter tracked persons by **age**, **gender**, and **ethnicity** to get targeted analytics.

## Files Created

- **[trait_filter.py](recog/trait_filter.py)** - Core filtering engine
- **[web_api.py](recog/web_api.py)** (updated) - Added filtering endpoints
- **[FILTERING_GUIDE.md](FILTERING_GUIDE.md)** - Complete documentation

## Quick Usage

### Option 1: Interactive Command-Line

```bash
python3 recog/trait_filter.py
```

Select filters interactively and export results.

### Option 2: Python API

```python
from trait_filter import TraitFilter

filter_system = TraitFilter()

# Filter by age and gender
results = filter_system.filter_by_traits(
    age_ranges=["(25-32)", "(38-43)"],
    genders=["Male"]
)

print(f"Found {len(results)} persons matching criteria")
```

### Option 3: Web API

```bash
# Start API server
python3 recog/web_api.py

# Filter via HTTP
curl "http://localhost:5000/api/filter/persons?age=(25-32)&gender=Male"

# Get demographics
curl "http://localhost:5000/api/demographics/summary"
```

## Filter Categories

**Age:** `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`

**Gender:** `Male`, `Female`

**Ethnicity:** `African`, `Caucasian`, `East Asian`, `South Asian`, `Hispanic/Latino`, `Middle Eastern`

## Integration with Your Detection System

When you detect age/gender/ethnicity, save it:

```python
from trait_filter import save_person_traits

save_person_traits(
    person_dir="recog/output/person_000",
    age="(25-32)",
    gender="Male",
    ethnicity="Caucasian",
    age_confidence=0.85,
    gender_confidence=0.92,
    ethnicity_confidence=0.78
)
```

Then you can filter by these traits!

## New API Endpoints

**Filter persons:**
```
GET /api/filter/persons?age=(25-32)&gender=Male&ethnicity=Caucasian
```

**Get demographic summary:**
```
GET /api/demographics/summary
```

## Example Response

```json
{
  "filtered_count": 5,
  "total_dwell_time": 1523.5,
  "average_dwell_time": 304.7,
  "total_visits": 12,
  "persons": [...],
  "demographic_summary": {
    "age_distribution": {"(25-32)": 5},
    "gender_distribution": {"Male": 5},
    "ethnicity_distribution": {"Caucasian": 5}
  }
}
```

## Use Cases

- **Retail:** Track which demographics spend most time browsing
- **Marketing:** Compare engagement across age/gender groups
- **Security:** Monitor specific demographic patterns
- **Research:** Analyze crowd demographics

## Documentation

See **[FILTERING_GUIDE.md](FILTERING_GUIDE.md)** for complete documentation with examples!
