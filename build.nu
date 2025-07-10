#!/usr/bin/env nu

# Build all typst files in 'src/' to their corresponding pdf files in 'out/'
def main [] {
  # Remove this to avoid typst for setting the current date to 1980/1/1
  if "SOURCE_DATE_EPOCH" in $env {
    hide-env SOURCE_DATE_EPOCH
  }

  ls src/*.typ | each { |file|
    let base = $file.name | path parse
    let out_pdf = $"out/($base.stem).pdf"

    print $"Compiling ($file.name) -> ($out_pdf)"
    typst compile --font-path src/template/fonts/ $file.name $out_pdf
  }

  null
}
