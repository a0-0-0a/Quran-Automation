//! Floor material = a small text file. Swap the floor (marble, wood, concrete...) without touching code.
//! Each material has its own texture zip, tiling size and reflection behaviour.

pub struct Material {
    pub name: String,
    pub texture: String,    // zip containing a .png albedo (any power-of-two size)
    pub tile_m: f32,        // metres covered by one texture tile
    pub albedo_gain: f32,   // brightness of the texture as diffuse colour
    pub f0: f32,            // reflection strength at normal incidence (dielectrics: 0.03-0.10)
    pub reflect_rough: f32, // sharpness of mirror reflections (text on floor); lower = more mirror-like
    pub gloss_rough: f32,   // roughness of the lamp highlight on the floor
    pub sharpen: f32,       // unsharp mask applied to the texture when loaded (0 = off)
    pub lod_bias: f32,      // <0 crisper texture, >0 softer
    pub vein_gloss: f32,    // how much the texture brightness varies the roughness (0 = uniform)
    pub env_gloss: f32,     // faint studio-light reflection in the polished floor (0 = off)
}

impl Material {
    pub fn green_marble() -> Material {
        Material {
            name: "green_marble".into(),
            texture: "textures.zip".into(),
            tile_m: 1.4,
            albedo_gain: 2.0,
            f0: 0.08,
            reflect_rough: 0.05,
            gloss_rough: 0.32,
            sharpen: 0.0,
            lod_bias: 0.0,
            vein_gloss: 0.0,
            env_gloss: 0.0,
        }
    }

    pub fn load(path: &str) -> Result<Material, String> {
        let txt = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut m = Material::green_marble();
        m.name = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        // A texture folder uses `material.txt`; name the material after the folder.
        if m.name == "material" {
            if let Some(dir) = std::path::Path::new(path).parent().and_then(|p| p.file_name()) {
                m.name = dir.to_string_lossy().to_string();
            }
        }
        for line in txt.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else { continue };
            let (k, v) = (k.trim(), v.trim());
            let f = || -> Result<f32, String> { v.parse::<f32>().map_err(|_| format!("{path}: bad number for '{k}'")) };
            match k {
                "texture" => m.texture = v.to_string(),
                "tile_m" => m.tile_m = f()?,
                "albedo_gain" => m.albedo_gain = f()?,
                "f0" => m.f0 = f()?,
                "reflect_rough" => m.reflect_rough = f()?,
                "gloss_rough" => m.gloss_rough = f()?,
                "sharpen" => m.sharpen = f()?,
                "lod_bias" => m.lod_bias = f()?,
                "vein_gloss" => m.vein_gloss = f()?,
                "env_gloss" => m.env_gloss = f()?,
                other => return Err(format!("{path}: unknown key '{other}'")),
            }
        }
        Ok(m)
    }
}
