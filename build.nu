#!/usr/bin/env nu

def main [] {
  hide-env SOURCE_DATE_EPOCH

  ls src/*.typ | each { |file|
    let base = $file.name | path parse
    let out_pdf = $"out/($base.stem).pdf"

    print $"Compiling ($file.name) -> ($out_pdf)"
    typst compile --font-path src/template/fonts/ $file.name $out_pdf
  }

  return
}
