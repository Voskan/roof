
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
import numpy as np
from typing import List, Dict, Union, Any

def export_to_geojson(polygons: List[np.ndarray], 
                      attributes: List[Dict[str, Any]], 
                      output_path: str, 
                      transform: rasterio.transform.Affine, 
                      crs: Union[str, rasterio.crs.CRS]):
    """
    Export polygons and attributes to a GeoJSON file.
    
    Args:
        polygons (List[np.ndarray]): List of polygon coordinates in pixel space (N, 1, 2) or (N, 2).
                                     Coordinates are (x, y) i.e., (col, row).
        attributes (List[Dict]): List of attribute dictionaries (e.g., {'slope': 15, 'azimuth': 90}).
        output_path (str): Path to save the GeoJSON file.
        transform (Affine): Affine transform from pixel space to map space (from rasterio src.transform).
        crs (str or CRS): Coordinate Reference System of the map space.
    """
    geometries = []
    valid_data = []
    
    if len(polygons) != len(attributes):
        print(f"Warning: Mismatch in polygons ({len(polygons)}) and attributes ({len(attributes)}) counts.")
        min_len = min(len(polygons), len(attributes))
        polygons = polygons[:min_len]
        attributes = attributes[:min_len]
    
    for i, (poly, attr) in enumerate(zip(polygons, attributes)):
        pts = poly.reshape(-1, 2)
        
        if len(pts) < 3:
            continue
            
        # Transform pixel coordinates (x=col, y=row) to map coordinates
        # rasterio transform: x_map, y_map = transform * (col, row)
        # We assume vertices are (x, y) pixels.
        
        map_pts = []
        for pt in pts:
            x_pix, y_pix = pt[0], pt[1]
            # Apply affine transform
            x_map, y_map = transform * (x_pix, y_pix)
            map_pts.append((x_map, y_map))
            
        # Create Shapely Polygon
        try:
            geom = Polygon(map_pts)
            if not geom.is_valid:
                geom = geom.buffer(0) # Fix invalid geometries
        except Exception as e:
            print(f"Error creating polygon {i}: {e}")
            continue
            
        geometries.append(geom)
        valid_data.append(attr)
    
    if not geometries:
        print("No valid geometries to export.")
        return
        
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(valid_data, geometry=geometries, crs=crs)
    
    # Save to file
    try:
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"Successfully exported {len(gdf)} features to {output_path}")
    except Exception as e:
        print(f"Failed to save GeoJSON: {e}")
