#!/usr/bin/env python3
"""
Build Label-Centric Knowledge Graph from UMLS + MeSH

This script builds a knowledge graph showing relationships between MeSH codes
within biomedical abstracts, enabling spurious correlation detection.

Author: Generated for biomedical NLP research
"""

import argparse
import json
import logging
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import csv

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeSHMapper:
    """Handles MeSH code metadata from Excel file."""
    
    def __init__(self, mesh_xlsx_path: str):
        """Load MeSH metadata from Excel."""
        logger.info(f"Loading MeSH metadata from {mesh_xlsx_path}")
        self.df = pd.read_excel(mesh_xlsx_path)
        
        # Build lookup dictionaries
        self.mesh_to_name = {}
        self.mesh_to_trees = {}
        
        for _, row in self.df.iterrows():
            mesh_id = row['Unique ID']
            name = row['MeSH Heading']
            tree_nums = row['Tree Number(s)']
            
            self.mesh_to_name[mesh_id] = name
            
            # Handle multiple tree numbers separated by commas
            if pd.notna(tree_nums):
                trees = [t.strip() for t in str(tree_nums).split(',')]
                self.mesh_to_trees[mesh_id] = trees
            else:
                self.mesh_to_trees[mesh_id] = []
        
        logger.info(f"Loaded {len(self.mesh_to_name)} MeSH codes")
    
    def get_name(self, mesh_id: str) -> Optional[str]:
        """Get preferred name for MeSH code."""
        return self.mesh_to_name.get(mesh_id)
    
    def get_trees(self, mesh_id: str) -> List[str]:
        """Get tree numbers for MeSH code."""
        return self.mesh_to_trees.get(mesh_id, [])
    
    def has_mesh(self, mesh_id: str) -> bool:
        """Check if MeSH code exists in metadata."""
        return mesh_id in self.mesh_to_name


class AbstractDatabase:
    """Handles loading and querying the neurology database."""
    
    def __init__(self, json_path: str, pmid_xlsx_path: Optional[str] = None, limit: Optional[int] = None):
        """Load abstract database and optionally filter by PMIDs."""
        logger.info(f"Loading abstract database from {json_path}")
        
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(raw_data, list):
            # Format: [{pmid: ..., ...}, {pmid: ..., ...}]
            data = raw_data
        elif isinstance(raw_data, dict):
            # Check for common wrapper keys
            if 'articles' in raw_data and isinstance(raw_data['articles'], list):
                data = raw_data['articles']
            elif 'data' in raw_data and isinstance(raw_data['data'], list):
                data = raw_data['data']
            else:
                # Assume keys are PMIDs or article IDs, values are article data
                data = list(raw_data.values())
                # But check if the first value is a list (likely the actual data)
                if data and isinstance(data[0], list):
                    data = data[0]
        else:
            raise TypeError(f"Unexpected JSON format: {type(raw_data)}")
        
        if not isinstance(data, list):
            raise TypeError(f"Could not extract article list from JSON. Got: {type(data)}")
        
        # Filter by PMIDs if provided
        if pmid_xlsx_path:
            pmid_df = pd.read_excel(pmid_xlsx_path)
            target_pmids = set(str(p) for p in pmid_df['PMID'].unique())
            data = [d for d in data if str(d['pmid']) in target_pmids]
            logger.info(f"Filtered to {len(data)} abstracts from PMID file")
        
        # Apply limit if specified
        if limit and isinstance(limit, int):
            data = list(data)[:int(limit)]
            logger.info(f"Limited to first {limit} abstracts")
        
        self.data = data
        logger.info(f"Loaded {len(self.data)} abstracts")
    
    def get_pmids(self) -> List[str]:
        """Get all PMIDs in database."""
        return [str(d['pmid']) for d in self.data]
    
    def get_mesh_for_pmid(self, pmid: str) -> List[str]:
        """Extract MeSH codes for a given PMID."""
        for article in self.data:
            if str(article['pmid']) == pmid:
                mesh_codes = []
                if 'meshMajorEnhanced' in article:
                    for entry in article['meshMajorEnhanced']:
                        # Extract unique_id_X fields
                        for key, value in entry.items():
                            if key.startswith('unique_id_') and value:
                                mesh_codes.append(value)
                return list(set(mesh_codes))  # Remove duplicates
        return []
    
    def get_all_mesh_codes(self) -> Set[str]:
        """Extract all unique MeSH codes from database."""
        all_mesh = set()
        for pmid in self.get_pmids():
            all_mesh.update(self.get_mesh_for_pmid(pmid))
        return all_mesh


class UMLSMapper:
    """Handles UMLS data loading and MeSH->CUI mapping."""
    
    def __init__(self, umls_dir: str):
        """Initialize with UMLS directory path."""
        self.umls_dir = Path(umls_dir)
        self.mrconso_path = self.umls_dir / 'MRCONSO.RRF'
        self.mrsty_path = self.umls_dir / 'MRSTY.RRF'
        self.mrrel_path = self.umls_dir / 'MRREL.RRF'
        
        # Verify files exist
        for path in [self.mrconso_path, self.mrsty_path, self.mrrel_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")
        
        self.cui_to_semtypes = {}
        self.mesh_to_cui = {}
        self.cui_to_synonyms = defaultdict(list)
        self.cui_to_pref_name = {}
    
    def load_semantic_types(self):
        """Load semantic types from MRSTY.RRF."""
        logger.info("Loading semantic types from MRSTY.RRF")
        self.cui_to_semtypes = defaultdict(list)
        
        with open(self.mrsty_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading MRSTY"), 1):
                # Skip header if present
                if line_num == 1 and line.startswith('CUI'):
                    continue
                    
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    cui = parts[0]
                    sty = parts[3]  # Semantic type name
                    self.cui_to_semtypes[cui].append(sty)
        
        logger.info(f"Loaded semantic types for {len(self.cui_to_semtypes)} CUIs")
    
    def map_mesh_to_cui(self, mesh_codes: Set[str], mesh_mapper: MeSHMapper) -> Dict[str, str]:
        """
        Map MeSH codes to CUIs using MRCONSO.RRF.
        
        Returns dict: {mesh_id: cui}
        """
        logger.info(f"Mapping {len(mesh_codes)} MeSH codes to CUIs")
        logger.info(f"Sample MeSH codes: {list(mesh_codes)[:5]}")
        
        # First pass: collect candidates
        mesh_candidates = defaultdict(list)
        lines_read = 0
        msh_lines = 0
        
        with open(self.mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Scanning MRCONSO (pass 1)"):
                lines_read += 1
                
                # Skip header row if present
                if lines_read == 1 and line.startswith('CUI'):
                    logger.info("Skipping header row in MRCONSO.RRF")
                    continue
                
                parts = line.strip().split('|')
                if len(parts) < 14:
                    continue
                
                cui = parts[0]
                lat = parts[1]
                sab = parts[11]
                
                # Filter: English, MeSH source
                if lat != 'ENG' or sab != 'MSH':
                    continue
                
                msh_lines += 1
                
                tty = parts[12]
                str_text = parts[14]
                ispref = parts[6]
                sdui = parts[10]  # Source descriptor ID (MeSH ID)
                
                # Check if this MeSH code is in our target set
                if sdui in mesh_codes:
                    mesh_candidates[sdui].append({
                        'cui': cui,
                        'str': str_text,
                        'ispref': ispref,
                        'tty': tty
                    })
        
        logger.info(f"Read {lines_read} total lines, {msh_lines} MSH lines")
        logger.info(f"Found candidates for {len(mesh_candidates)} MeSH codes")
        logger.info(f"Sample candidates: {dict(list(mesh_candidates.items())[:2])}")
        
        # Second pass: tie-breaking
        mesh_to_cui = {}
        
        for mesh_code in mesh_codes:
            candidates = mesh_candidates.get(mesh_code, [])
            
            if not candidates:
                logger.warning(f"No UMLS mapping found for MeSH code: {mesh_code}")
                continue
            
            # Tie-break priority 1: ISPREF = 'Y'
            pref_candidates = [c for c in candidates if c['ispref'] == 'Y']
            if pref_candidates:
                candidates = pref_candidates
            
            if len(candidates) == 1:
                mesh_to_cui[mesh_code] = candidates[0]['cui']
                continue
            
            # Tie-break priority 2: TTY in {'MH', 'NM'}
            tty_candidates = [c for c in candidates if c['tty'] in {'MH', 'NM'}]
            if tty_candidates:
                candidates = tty_candidates
            
            if len(candidates) == 1:
                mesh_to_cui[mesh_code] = candidates[0]['cui']
                continue
            
            # Tie-break priority 3: Exact string match with MeSH preferred name
            mesh_pref_name = mesh_mapper.get_name(mesh_code)
            if mesh_pref_name:
                exact_matches = [c for c in candidates if c['str'] == mesh_pref_name]
                if exact_matches:
                    candidates = exact_matches
            
            if len(candidates) == 1:
                mesh_to_cui[mesh_code] = candidates[0]['cui']
                continue
            
            # Tie-break priority 4: Prefer CUIs with semantic types
            candidates_with_semtypes = [c for c in candidates 
                                       if c['cui'] in self.cui_to_semtypes]
            if candidates_with_semtypes:
                candidates = candidates_with_semtypes
            
            # Final: pick first (deterministic)
            selected = candidates[0]
            mesh_to_cui[mesh_code] = selected['cui']
            
            if len(candidates) > 1:
                logger.warning(f"Multiple CUIs for {mesh_code}, selected {selected['cui']}")
        
        logger.info(f"Mapped {len(mesh_to_cui)} MeSH codes to CUIs")
        self.mesh_to_cui = mesh_to_cui
        return mesh_to_cui
    
    def load_cui_metadata(self, cuis: Set[str]):
        """Load synonyms and preferred names for CUIs."""
        logger.info(f"Loading metadata for {len(cuis)} CUIs")
        
        self.cui_to_synonyms = defaultdict(list)
        self.cui_to_pref_name = {}
        
        with open(self.mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading CUI metadata"), 1):
                # Skip header if present
                if line_num == 1 and line.startswith('CUI'):
                    continue
                    
                parts = line.strip().split('|')
                if len(parts) < 15:
                    continue
                
                cui = parts[0]
                if cui not in cuis:
                    continue
                
                lat = parts[1]
                str_text = parts[14]
                ispref = parts[6]
                
                if lat == 'ENG':
                    self.cui_to_synonyms[cui].append(str_text)
                    
                    # Store preferred name
                    if ispref == 'Y' and cui not in self.cui_to_pref_name:
                        self.cui_to_pref_name[cui] = str_text


class KnowledgeGraphBuilder:
    """Builds knowledge graph from UMLS relationships."""
    
    def __init__(self, umls_mapper: UMLSMapper):
        """Initialize with UMLS mapper."""
        self.umls_mapper = umls_mapper
        self.edges = []
        self.relationship_types = Counter()
    
    def build_edges(self, pmid_to_cuis: Dict[str, List[str]]):
        """
        Build edges between CUIs that appear in the same abstract.
        
        Args:
            pmid_to_cuis: Dict mapping PMID to list of CUIs in that abstract
        """
        logger.info("Building edges from MRREL.RRF")
        
        # Create reverse index: cui -> set of PMIDs it appears in
        cui_to_pmids = defaultdict(set)
        for pmid, cuis in pmid_to_cuis.items():
            for cui in cuis:
                cui_to_pmids[cui].add(pmid)
        
        # Track edges per PMID
        pmid_edges = defaultdict(list)
        all_cuis = set(cui for cuis in pmid_to_cuis.values() for cui in cuis)
        
        logger.info(f"Scanning MRREL for relationships among {len(all_cuis)} CUIs")
        
        with open(self.umls_mapper.mrrel_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(tqdm(f, desc="Scanning MRREL"), 1):
                # Skip header if present
                if line_num == 1 and (line.startswith('CUI') or 'CUI1' in line):
                    continue
                    
                parts = line.strip().split('|')
                if len(parts) < 8:
                    continue
                
                cui1 = parts[0]
                cui2 = parts[4]
                rel = parts[3]
                rela = parts[7] if parts[7] else None
                
                # Skip self-loops (same CUI)
                if cui1 == cui2:
                    continue
                
                # Skip synonym relationships (SY) - these are within-concept, not between-concept
                if rel == 'SY':
                    continue
                
                # Only consider if both CUIs are in our graph
                if cui1 not in all_cuis or cui2 not in all_cuis:
                    continue
                
                # Find PMIDs where both CUIs appear together
                pmids1 = cui_to_pmids[cui1]
                pmids2 = cui_to_pmids[cui2]
                common_pmids = pmids1 & pmids2
                
                if not common_pmids:
                    continue
                
                # Create edge
                rel_type = f"{rel}:{rela}" if rela else rel
                
                for pmid in common_pmids:
                    edge = {
                        'cui1': cui1,
                        'cui2': cui2,
                        'rel': rel,
                        'rela': rela,
                        'rel_type': rel_type,
                        'pmid': pmid
                    }
                    pmid_edges[pmid].append(edge)
                    self.relationship_types[rel_type] += 1
        
        # Store edges
        for pmid, edges in pmid_edges.items():
            self.edges.extend(edges)
        
        logger.info(f"Found {len(self.edges)} edges across {len(pmid_edges)} PMIDs")
        logger.info(f"Found {len(self.relationship_types)} unique relationship types")


class SpuriousDetector:
    """Detects spurious (unconnected) MeSH codes in abstracts."""
    
    def __init__(self, edges: List[Dict], pmid_to_mesh: Dict[str, List[str]], 
                 mesh_to_cui: Dict[str, str]):
        """Initialize detector."""
        self.edges = edges
        self.pmid_to_mesh = pmid_to_mesh
        self.mesh_to_cui = mesh_to_cui
        self.cui_to_mesh = {v: k for k, v in mesh_to_cui.items()}
    
    def detect(self) -> Dict[str, Dict]:
        """
        Detect spurious MeSH codes for each PMID.
        
        Returns:
            Dict mapping PMID to spurious detection results
        """
        logger.info("Detecting spurious MeSH codes")
        
        results = {}
        
        for pmid, mesh_codes in tqdm(self.pmid_to_mesh.items(), desc="Detecting spurious"):
            # Get CUIs for this PMID
            cuis = [self.mesh_to_cui[m] for m in mesh_codes if m in self.mesh_to_cui]
            cui_set = set(cuis)
            
            # Find edges for this PMID
            pmid_edges = [e for e in self.edges if e['pmid'] == pmid]
            
            # Check connectivity for each MeSH code
            mesh_connectivity = {}
            for mesh_code in mesh_codes:
                cui = self.mesh_to_cui.get(mesh_code)
                if not cui:
                    mesh_connectivity[mesh_code] = {
                        'cui': None,
                        'is_spurious': True,
                        'connected_to': [],
                        'reason': 'No CUI mapping'
                    }
                    continue
                
                # Find connections
                connected_cuis = set()
                for edge in pmid_edges:
                    if edge['cui1'] == cui and edge['cui2'] in cui_set:
                        connected_cuis.add(edge['cui2'])
                    elif edge['cui2'] == cui and edge['cui1'] in cui_set:
                        connected_cuis.add(edge['cui1'])
                
                # Convert back to MeSH codes
                connected_mesh = [self.cui_to_mesh.get(c, c) for c in connected_cuis]
                
                mesh_connectivity[mesh_code] = {
                    'cui': cui,
                    'is_spurious': len(connected_cuis) == 0,
                    'connected_to': connected_mesh,
                    'num_connections': len(connected_cuis)
                }
            
            # Summary
            spurious_codes = [m for m, info in mesh_connectivity.items() 
                            if info['is_spurious']]
            connected_codes = [m for m, info in mesh_connectivity.items() 
                             if not info['is_spurious']]
            
            results[pmid] = {
                'total_mesh': len(mesh_codes),
                'connected_mesh': len(connected_codes),
                'spurious_mesh': len(spurious_codes),
                'spurious_codes': spurious_codes,
                'connected_codes': connected_codes,
                'mesh_connectivity': mesh_connectivity
            }
        
        logger.info(f"Spurious detection complete for {len(results)} PMIDs")
        return results


def write_outputs(output_dir: Path, nodes_data: List[Dict], edges_data: List[Dict],
                 spurious_results: Dict, mesh_to_cui: Dict, cui_to_id: Dict,
                 relationship_types: Counter, missing_mesh: Set[str]):
    """Write all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. nodes.csv
    logger.info("Writing nodes.csv")
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv(output_dir / 'nodes.csv', index=False)
    
    # 2. edges.csv
    logger.info("Writing edges.csv")
    if edges_data:
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(output_dir / 'edges.csv', index=False)
    else:
        # Write empty file with headers
        with open(output_dir / 'edges.csv', 'w') as f:
            f.write("src_id,dst_id,rel,rela,rel_type,pmid,weight\n")
    
    # 3. abstract_mesh.csv
    logger.info("Writing abstract_mesh.csv")
    abstract_mesh_rows = []
    for pmid, result in spurious_results.items():
        for mesh_code, info in result['mesh_connectivity'].items():
            abstract_mesh_rows.append({
                'pmid': pmid,
                'mesh_id': mesh_code,
                'cui': info['cui'],
                'int_id': cui_to_id.get(info['cui']),
                'is_spurious': info['is_spurious'],
                'num_connections': info.get('num_connections', 0),
                'connected_to_mesh_ids': ';'.join(info['connected_to'])
            })
    
    abstract_mesh_df = pd.DataFrame(abstract_mesh_rows)
    abstract_mesh_df.to_csv(output_dir / 'abstract_mesh.csv', index=False)
    
    # 4. spurious_summary.json
    logger.info("Writing spurious_summary.json")
    summary = {}
    for pmid, result in spurious_results.items():
        summary[pmid] = {
            'total_mesh': result['total_mesh'],
            'connected_mesh': result['connected_mesh'],
            'spurious_mesh': result['spurious_mesh'],
            'spurious_codes': result['spurious_codes'],
            'connected_codes': result['connected_codes']
        }
    
    with open(output_dir / 'spurious_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 5. relationship_types.json
    logger.info("Writing relationship_types.json")
    rel_data = {
        'total_relationship_types': len(relationship_types),
        'total_edges': sum(relationship_types.values()),
        'relationships': [
            {
                'rel_type': rel_type,
                'count': count,
                'percentage': round(100 * count / sum(relationship_types.values()), 2)
            }
            for rel_type, count in relationship_types.most_common()
        ]
    }
    
    with open(output_dir / 'relationship_types.json', 'w') as f:
        json.dump(rel_data, f, indent=2)
    
    # 6. maps.json
    logger.info("Writing maps.json")
    id_to_cui = {v: k for k, v in cui_to_id.items()}
    maps = {
        'cui_to_id': cui_to_id,
        'id_to_cui': id_to_cui,
        'mesh_to_cui': mesh_to_cui
    }
    
    with open(output_dir / 'maps.json', 'w') as f:
        json.dump(maps, f, indent=2)
    
    # 7. missing_mesh_codes.txt
    if missing_mesh:
        logger.info(f"Writing missing_mesh_codes.txt ({len(missing_mesh)} codes)")
        with open(output_dir / 'missing_mesh_codes.txt', 'w') as f:
            for mesh_code in sorted(missing_mesh):
                f.write(f"{mesh_code}\n")


def print_summary(nodes_data: List[Dict], edges_data: List[Dict], 
                 spurious_results: Dict, relationship_types: Counter):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*60)
    
    print(f"\nNodes:")
    print(f"  Total nodes: {len(nodes_data)}")
    print(f"  Unique CUIs: {len(set(n['cui'] for n in nodes_data))}")
    
    print(f"\nEdges:")
    print(f"  Total edges: {len(edges_data)}")
    print(f"  Unique relationship types: {len(relationship_types)}")
    
    print(f"\nTop 10 Relationship Types:")
    for rel_type, count in relationship_types.most_common(10):
        print(f"  {rel_type}: {count}")
    
    print(f"\nSpurious Detection:")
    total_mesh = sum(r['total_mesh'] for r in spurious_results.values())
    total_spurious = sum(r['spurious_mesh'] for r in spurious_results.values())
    total_connected = sum(r['connected_mesh'] for r in spurious_results.values())
    
    print(f"  Total MeSH codes: {total_mesh}")
    print(f"  Connected: {total_connected} ({100*total_connected/total_mesh:.1f}%)")
    print(f"  Spurious: {total_spurious} ({100*total_spurious/total_mesh:.1f}%)")
    
    print(f"\nPer-Abstract Statistics:")
    for i, (pmid, result) in enumerate(list(spurious_results.items())[:5]):
        print(f"\n  PMID {pmid}:")
        print(f"    Total MeSH: {result['total_mesh']}")
        print(f"    Connected: {result['connected_mesh']}")
        print(f"    Spurious: {result['spurious_mesh']}")
        if result['spurious_codes']:
            print(f"    Spurious codes: {', '.join(result['spurious_codes'][:3])}")
    
    print("\n" + "="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Build label-centric knowledge graph from UMLS + MeSH'
    )
    parser.add_argument('--umls-dir', required=True,
                       help='Path to UMLS META directory containing RRF files')
    parser.add_argument('--mesh-xlsx', required=True,
                       help='Path to MeSH codes Excel file')
    parser.add_argument('--database-json', required=True,
                       help='Path to neurology database JSON file')
    parser.add_argument('--pmid-xlsx', required=False,
                       help='Path to PMID Excel file (optional)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of PMIDs to process (default: all)')
    
    args = parser.parse_args()
    
    try:
        # Phase 0: Load MeSH metadata
        mesh_mapper = MeSHMapper(args.mesh_xlsx)
        
        # Phase 1: Load abstract database
        db = AbstractDatabase(args.database_json, args.pmid_xlsx, args.limit)
        pmids = db.get_pmids()
        logger.info(f"Processing {len(pmids)} PMIDs")
        
        # Extract MeSH codes per PMID
        pmid_to_mesh = {}
        all_mesh_codes = set()
        for pmid in pmids:
            mesh_codes = db.get_mesh_for_pmid(pmid)
            pmid_to_mesh[pmid] = mesh_codes
            all_mesh_codes.update(mesh_codes)
        
        logger.info(f"Found {len(all_mesh_codes)} unique MeSH codes")
        
        # Check for missing MeSH codes
        missing_mesh = set()
        for mesh_code in all_mesh_codes:
            if not mesh_mapper.has_mesh(mesh_code):
                missing_mesh.add(mesh_code)
        
        if missing_mesh:
            logger.warning(f"Found {len(missing_mesh)} MeSH codes not in meshcodes.xlsx")
        
        # Phase 2: Load UMLS data
        umls_mapper = UMLSMapper(args.umls_dir)
        umls_mapper.load_semantic_types()
        
        # Phase 3: Map MeSH to CUIs
        mesh_to_cui = umls_mapper.map_mesh_to_cui(all_mesh_codes, mesh_mapper)
        
        # Get all CUIs
        all_cuis = set(mesh_to_cui.values())
        umls_mapper.load_cui_metadata(all_cuis)
        
        # Phase 4: Build nodes
        logger.info("Building nodes")
        nodes_data = []
        cui_to_id = {}
        
        for idx, (mesh_code, cui) in enumerate(sorted(mesh_to_cui.items())):
            synonyms = umls_mapper.cui_to_synonyms.get(cui, [])
            semtypes = umls_mapper.cui_to_semtypes.get(cui, [])
            pref_name = umls_mapper.cui_to_pref_name.get(cui, mesh_mapper.get_name(mesh_code))
            tree_nums = mesh_mapper.get_trees(mesh_code)
            
            # Count PMIDs this MeSH appears in
            pmid_count = sum(1 for mesh_list in pmid_to_mesh.values() if mesh_code in mesh_list)
            
            node = {
                'int_id': idx,
                'cui': cui,
                'mesh_id': mesh_code,
                'pref_name': pref_name,
                'semtypes_json': json.dumps(semtypes),
                'treenums_json': json.dumps(tree_nums),
                'syn_count': len(synonyms),
                'pmid_count': pmid_count
            }
            
            nodes_data.append(node)
            cui_to_id[cui] = idx
        
        # Phase 5: Build edges
        pmid_to_cuis = {}
        for pmid, mesh_codes in pmid_to_mesh.items():
            cuis = [mesh_to_cui[m] for m in mesh_codes if m in mesh_to_cui]
            pmid_to_cuis[pmid] = cuis
        
        kg_builder = KnowledgeGraphBuilder(umls_mapper)
        kg_builder.build_edges(pmid_to_cuis)
        
        # Convert edges to output format
        edges_data = []
        for edge in kg_builder.edges:
            edges_data.append({
                'src_id': cui_to_id.get(edge['cui1']),
                'dst_id': cui_to_id.get(edge['cui2']),
                'rel': edge['rel'],
                'rela': edge['rela'],
                'rel_type': edge['rel_type'],
                'pmid': edge['pmid'],
                'weight': 1.0
            })
        
        # Phase 6: Spurious detection
        detector = SpuriousDetector(kg_builder.edges, pmid_to_mesh, mesh_to_cui)
        spurious_results = detector.detect()
        
        # Phase 7: Write outputs
        output_dir = Path(args.output_dir)
        write_outputs(output_dir, nodes_data, edges_data, spurious_results,
                     mesh_to_cui, cui_to_id, kg_builder.relationship_types,
                     missing_mesh)
        
        # Phase 8: Print summary
        print_summary(nodes_data, edges_data, spurious_results,
                     kg_builder.relationship_types)
        
        logger.info(f"✓ All outputs written to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
