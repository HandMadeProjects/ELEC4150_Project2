import pymupdf, os, sys

out = r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2\project-info\pages"
os.makedirs(out, exist_ok=True)

base = r"C:\Users\Atharva Pawar\Documents\GitHub\ELEC4150_Project2\ELEC4150_Project2\project-info"

rubric = os.path.join(base, "Projec_2_Rubric.pdf")
spec   = os.path.join(base, "Projec_2_Specification.pdf")

doc = pymupdf.open(rubric)
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=180)
    path = os.path.join(out, f"rubric_p{i+1}.png")
    pix.save(path)
    sys.stdout.write(f"Saved {path}\n"); sys.stdout.flush()
doc.close()

doc2 = pymupdf.open(spec)
for i, page in enumerate(doc2):
    pix = page.get_pixmap(dpi=150)
    path = os.path.join(out, f"spec_p{i+1}.png")
    pix.save(path)
    sys.stdout.write(f"Saved {path}\n"); sys.stdout.flush()
doc2.close()

sys.stdout.write("ALL DONE\n"); sys.stdout.flush()
