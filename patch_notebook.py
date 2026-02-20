import nbformat

nb_path = "notebooks/checkpoint_inference_test.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        source = cell.source
        
        # Patch Vectorization
        if "cv2.approxPolyDP" in source and "cv2.polylines(overlay" in source:
            new_source = """# ======================== OVERLAY WITH SLOPE LABELS ========================
from deeproof.utils.vectorization import regularize_building_polygons

overlay = cv2.addWeighted(img_rgb, 0.55, sem_vis, 0.45, 0.0)

roof_polygons = []
for i, inst in enumerate(instances):
    reg_polygons = regularize_building_polygons(
        inst['mask'].astype(np.uint8), 
        epsilon_factor=0.04, 
        ortho_threshold=10.0, 
        min_area=50
    )
    
    for poly in reg_polygons:
        poly_2d = poly.reshape(-1, 2)
        if poly_2d.shape[0] < 3:
            continue
            
        area = float(cv2.contourArea(poly))
        pts = poly_2d.reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

        m = cv2.moments(poly)
        if m['m00'] > 0:
            cx, cy = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
        else:
            cx, cy = int(poly_2d[0, 0]), int(poly_2d[0, 1])

        if inst['slope_deg'] is not None and area > 200:
            txt = f"{inst['slope_deg']:.0f}deg"
            cv2.putText(overlay, txt, (cx-10, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

        roof_polygons.append({
            'polygon_id': len(roof_polygons),
            'instance_id': i,
            'query_id': inst['query_id'],
            'class_id': inst['class_id'],
            'class_name': inst['class_name'],
            'score': inst['score'],
            'area_px': area,
            'slope_deg': inst['slope_deg'],
            'azimuth_deg': inst['azimuth_deg'],
            'normal': inst['normal'],
            'points_xy': poly_2d.astype(int).tolist(),
        })

cv2.imwrite(str(OUTPUT_DIR / 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUTPUT_DIR / 'semantic_mask.png'), sem_map)

plt.figure(figsize=(12, 12))
plt.imshow(overlay)
plt.title(f'{len(roof_polygons)} regularized roof polygons with slope angles')
plt.axis('off')
plt.show()
print(f'Total polygons: {len(roof_polygons)}')"""
            cell.source = new_source

        # Patch Interactive 3D Roof Model to use regularize_building_polygons
        if "cv2.approxPolyDP" in source and "zs_roof =" in source and "fig3d.add_trace" in source:
            old_str = """    contours, _ = cv2.findContours(
        inst['mask'].astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:
            continue

        epsilon = 0.006 * cv2.arcLength(contour, True)
        poly_2d = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
        if poly_2d.shape[0] < 3:
            continue

        xs = poly_2d[:, 0]
        ys = poly_2d[:, 1]"""
            
            new_str = """    try:
        from deeproof.utils.vectorization import regularize_building_polygons
    except ImportError:
        pass

    reg_polygons = regularize_building_polygons(
        inst['mask'].astype(np.uint8), 
        epsilon_factor=0.04, 
        ortho_threshold=10.0, 
        min_area=200
    )

    for poly in reg_polygons:
        poly_2d = poly.reshape(-1, 2).astype(float)
        if poly_2d.shape[0] < 3:
            continue

        xs = poly_2d[:, 0]
        ys = poly_2d[:, 1]"""
            cell.source = source.replace(old_str, new_str)
            
        # Patch OBJ export
        if "cv2.approxPolyDP" in source and "f.write(f'vn " in source:
            old_str2 = """        contours, _ = cv2.findContours(
            inst['mask'].astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            epsilon = 0.01 * cv2.arcLength(contour, True)
            poly_2d = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if poly_2d.shape[0] < 3:
                continue"""
                
            new_str2 = """        try:
            from deeproof.utils.vectorization import regularize_building_polygons
        except ImportError:
            pass

        reg_polygons = regularize_building_polygons(
            inst['mask'].astype(np.uint8), 
            epsilon_factor=0.04, 
            ortho_threshold=10.0, 
            min_area=100
        )
        for poly in reg_polygons:
            poly_2d = poly.reshape(-1, 2)
            if poly_2d.shape[0] < 3:
                continue"""
            cell.source = cell.source.replace(old_str2, new_str2)

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

