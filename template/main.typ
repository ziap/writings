#let template(
  title: "",
  subtitle: none,
  author: "",
  affiliation: none,
  other: none,
  toc: false,
  date: datetime.today().display("[month repr:long] [day], [year]"),
  logo: none,
  main-color: "E94845",
  alpha: 60%,
  color-words: (),
  body,
) = {
  set document(author: author, title: title)

  let body-font = "Source Sans 3"
  let title-font = "Merriweather Sans"

  let primary-color = rgb(main-color)
  let secondary-color = color.mix(color.rgb(100%, 100%, 100%, alpha), primary-color, space:rgb)

  set figure.caption(separator: [ --- ], position: top)
  show raw.where(block: false) : it => h(0.5em) + box(fill: primary-color.lighten(90%), outset: 0.2em, it) + h(0.5em)

  set text(font: body-font, 12pt)

  show heading: set text(font: title-font, fill: primary-color)
  show heading: it => it + v(0.5em)
  set heading(numbering: (..nums) => {
    let level = nums.pos().len()

    if level < 4 {
      numbering("1.", ..nums)
    }
  })

  set math.equation(numbering: "(1)")
  show link: it => underline(text(fill: primary-color, it))

  set enum(indent: 1em, numbering: n => [#text(fill: primary-color, numbering("1.", n))])
  set list(indent: 1em, marker: n => [#text(fill: primary-color, "•")])

  // Title page.

  // Logo at top right if given
  if logo != none {
    set image(width: 6cm)
    place(top + right, logo)
  }
  place(top + left, dx: -35%, dy: -28%, circle(radius: 150pt, fill: primary-color))
  place(top + left, dx: -10%, circle(radius: 75pt, fill: secondary-color))
  place(bottom +right, dx: 40%, dy: 30%, circle(radius: 150pt, fill: secondary-color))

  v(2fr)

  align(center, text(font: title-font, 2.5em, weight: 700, title))
  v(2em, weak: true)
  if subtitle != none {
    align(center, text(font: title-font, 2em, weight: 700, subtitle))
    v(2em, weak: true)
  }
  align(center, text(1.1em, date))

  v(2fr)

  align(center)[
    #if author != "" {
      strong(author)
      linebreak()
    }
    #if affiliation != none {
      affiliation
      linebreak()
    }
    #if other != none {
      emph(other.join(linebreak()))
      linebreak()
    }
  ]

  pagebreak()

  set page(
    paper: "a4",
    numbering: "1 / 1", 
    number-align: center, 
    header: [#emph()[#title #h(1fr) #author]]
  )

  counter(page).update(1)
  set par(justify: true)

  if toc {
    show outline.entry: it => text(size: 12pt, weight: "regular",it)
    set outline(title: "Table of contents")
    outline()
  }

  body
}
