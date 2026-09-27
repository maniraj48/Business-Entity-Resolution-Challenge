"""
Dataset generator script for local testing and pipeline verification.
Generates compliant TSV dataset files in dataset/train and dataset/test.
"""

import os
import pandas as pd

def generate_datasets():
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # TRAIN DATASET GENERATION
    # ----------------------------------------------------
    # Entities: 10 entities in S1
    # Matches:
    # S1-1001 -> S2-2001, S3-3001 (Acme Corp, US)
    # S1-1002 -> S2-2002 (Global Solutions Pvt Ltd, India)
    # S1-1003 -> S3-3003 (Bistro de Paris SARL, France)
    # S1-1004 -> S2-2004, S3-3004 (Apex Technologies Inc, US)
    # S1-1005 -> S2-2005 (TechVision Group, Germany)
    # S1-1006 -> (Singleton - No matches) (Sunrise Bakeries LLC, US)
    # S1-1007 -> S3-3007 (Pacific Traders Co, Australia)
    # S1-1008 -> S2-2008, S3-3008 (Zenith Logistics, Japan)
    # S1-1009 -> (Singleton - No matches) (Starlight Cafe, UK)
    # S1-1010 -> S2-2010 (Horizon Dynamics, Canada)
    
    train_s1 = [
        {"entity_id": "S1-1001", "business_name": "Acme Corporation", "business_address": "100 Main St, Suite 400, New York, NY", "country": "US"},
        {"entity_id": "S1-1002", "business_name": "Global Solutions Pvt. Ltd.", "business_address": "42 MG Road, Bangalore, Karnataka", "country": "India"},
        {"entity_id": "S1-1003", "business_name": "Bistro de Paris", "business_address": "15 Rue de la Paix, Paris", "country": "France"},
        {"entity_id": "S1-1004", "business_name": "Apex Technologies Inc.", "business_address": "500 Innovation Way, San Jose, CA", "country": "USA"},
        {"entity_id": "S1-1005", "business_name": "TechVision Group GmbH", "business_address": "Berliner Str. 12, Berlin", "country": "Germany"},
        {"entity_id": "S1-1006", "business_name": "Sunrise Bakeries LLC", "business_address": "12 Baker St, Boston, MA", "country": "US"},
        {"entity_id": "S1-1007", "business_name": "Pacific Traders Co.", "business_address": "88 George St, Sydney, NSW", "country": "Australia"},
        {"entity_id": "S1-1008", "business_name": "Zenith Logistics Inc", "business_address": "1-2-3 Chiyoda, Tokyo", "country": "Japan"},
        {"entity_id": "S1-1009", "business_name": "Starlight Cafe", "business_address": "10 Oxford Street, London", "country": "UK"},
        {"entity_id": "S1-1010", "business_name": "Horizon Dynamics Ltd", "business_address": "250 Bay St, Toronto, ON", "country": "Canada"},
    ]
    
    train_s2 = [
        {"entity_id": "S2-2001", "business_name": "Acme Corp", "business_address": "100 Main Street Ste 400, NY", "country": "United States"},
        {"entity_id": "S2-2002", "business_name": "Global Solutions Private Limited", "business_address": "42 M.G. Rd, Bangalore", "country": "IN"},
        {"entity_id": "S2-2004", "business_name": "Apex Tech Inc", "business_address": "500 Innovation Way, San Jose", "country": "US"},
        {"entity_id": "S2-2005", "business_name": "Tech Vision Group", "business_address": "Berlinerstrasse 12, Berlin", "country": "DEU"},
        {"entity_id": "S2-2008", "business_name": "Zenith Logistics", "business_address": "1-2-3 Chiyoda-ku, Tokyo", "country": "JP"},
        {"entity_id": "S2-2010", "business_name": "Horizon Dynamics Limited", "business_address": "250 Bay Street, Toronto", "country": "CA"},
        {"entity_id": "S2-2099", "business_name": "Unrelated Business Alpha", "business_address": "99 Random Rd, Dallas, TX", "country": "US"},
    ]
    
    train_s3 = [
        {"entity_id": "S3-3001", "business_name": "Acme Company Inc", "business_address": "100 Main St, New York", "country": "US"},
        {"entity_id": "S3-3003", "business_name": "Le Bistro de Paris", "business_address": "15 Rue de la Paix 75002 Paris", "country": "FR"},
        {"entity_id": "S3-3004", "business_name": "Apex Technologies", "business_address": "500 Innovation Way Suite 10, San Jose, CA", "country": "United States"},
        {"entity_id": "S3-3007", "business_name": "Pacific Traders", "business_address": "88 George Street, Sydney", "country": "AUS"},
        {"entity_id": "S3-3008", "business_name": "Zenith Logistics Co", "business_address": "Chiyoda 1-2-3, Tokyo", "country": "Japan"},
        {"entity_id": "S3-3099", "business_name": "Unrelated Business Beta", "business_address": "100 Pine St, Seattle, WA", "country": "US"},
    ]
    
    train_gt = [
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001,S3-3001"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": "S2-2002"},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": "S3-3003"},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": "S2-2004,S3-3004"},
        {"source1_entity_id": "S1-1005", "matched_entity_ids": "S2-2005"},
        {"source1_entity_id": "S1-1006", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1007", "matched_entity_ids": "S3-3007"},
        {"source1_entity_id": "S1-1008", "matched_entity_ids": "S2-2008,S3-3008"},
        {"source1_entity_id": "S1-1009", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-1010", "matched_entity_ids": "S2-2010"},
    ]
    
    # ----------------------------------------------------
    # TEST DATASET GENERATION
    # ----------------------------------------------------
    test_s1 = [
        {"entity_id": "S1-TEST-01", "business_name": "Vortex Analytics LLC", "business_address": "777 Silicon Alley, San Francisco, CA", "country": "US"},
        {"entity_id": "S1-TEST-02", "business_name": "Chateau Margaux Fine Wines", "business_address": "5 Route du Vin, Bordeaux", "country": "France"},
        {"entity_id": "S1-TEST-03", "business_name": "Bharat BioTech Services", "business_address": "12 Cyber City, Hyderabad", "country": "India"},
        {"entity_id": "S1-TEST-04", "business_name": "Northern Lights Expeditions", "business_address": "45 Fjord Way, Tromso", "country": "Norway"},
        {"entity_id": "S1-TEST-05", "business_name": "Omni Health Systems", "business_address": "300 Medical Center Dr, Chicago, IL", "country": "USA"},
    ]
    
    test_s2 = [
        {"entity_id": "S2-TEST-21", "business_name": "Vortex Analytics", "business_address": "777 Silicon Alley, SF, CA", "country": "United States"},
        {"entity_id": "S2-TEST-23", "business_name": "Bharat Biotech", "business_address": "12 Cyber City, Hyderabad, AP", "country": "IN"},
        {"entity_id": "S2-TEST-25", "business_name": "Omni Health Systems Inc", "business_address": "300 Medical Ctr Dr, Chicago", "country": "US"},
    ]
    
    test_s3 = [
        {"entity_id": "S3-TEST-31", "business_name": "Vortex Analytics Corporation", "business_address": "777 Silicon Alley, San Francisco", "country": "US"},
        {"entity_id": "S3-TEST-32", "business_name": "Chateau Margaux Wines", "business_address": "5 Route du Vin 33460 Margaux", "country": "France"},
        {"entity_id": "S3-TEST-34", "business_name": "Northern Lights Expeditions AS", "business_address": "45 Fjord Way, Tromso", "country": "Norway"},
    ]
    
    pd.DataFrame(train_s1).to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)
    pd.DataFrame(train_s2).to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)
    pd.DataFrame(train_s3).to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)
    pd.DataFrame(train_gt).to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)
    
    pd.DataFrame(test_s1).to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)
    pd.DataFrame(test_s2).to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)
    pd.DataFrame(test_s3).to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)
    
    print("[SUCCESS] Training and test datasets generated in dataset/train and dataset/test")

if __name__ == "__main__":
    generate_datasets()
