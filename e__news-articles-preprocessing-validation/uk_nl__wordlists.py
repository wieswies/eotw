#######################
######### NL ##########
#######################

nl23_political_words = {
    'PVV': {
        'multistring_matches': ['Partij voor de Vrijheid'],
        'substring_matches': [],
        'exact_matches': ['PVV', 'Wilders', 'Agema', 'Bosma', 'Graus']
    },
    'GL-PvdA': {
        'multistring_matches': ['Partij van de Arbeid', 'PvdA-GL', 'GL-PvdA', 'PvdA/GL', 'GL/PvdA', 'PvdA-GroenLinks', 'GroenLinks-PvdA',
                                'PvdA/GroenLinks', 'GroenLinks/PvdA'],
        'substring_matches': ['PvdA', 'GroenLinks'],
        'exact_matches': ['GL', 'Timmermans', 'Lahlah', 'Klaver'],
    },
    'VVD': {
        'multistring_matches': ['Volkspartij voor Vrijheid en Democratie', 'Minister van Leeuwen',
                                'van Leeuwen', 'van der Wal', 'van der Burg', 'van der Maat',
                                'Staatssecretaris de Vries', 'Mariëlle Paul'],
        'substring_matches': [],
        'exact_matches': ['VVD', 'Yesilgöz', 'Yesilgoz', 'Hermans', 'Rutte', ' Helder', 'Adriaansens', 'Harbers', 'Wiersma']
    },
    'NSC': {
        'multistring_matches': ['Nieuw Sociaal Contract', 'van Vroonhoven'],
        'substring_matches': [],
        'exact_matches': ['NSC', 'Omtzigt']
    },
    'D66': {
        'multistring_matches': ['Democraten 66', 'van Weyenberg', 'van Huffelen'],
        'substring_matches': [],
        'exact_matches': ['D66', 'Jetten', 'Paternotte', 'Kaag', 'Ollongren', 'Dijkgraaf', 'Kuipers', 'Dijkstra',
                          'Uslu', 'Weerwind', 'Gräper', 'Vijlbrief']
    },
    'BBB': {
        'multistring_matches': ['van der Plas'],
        'substring_matches': [],
        'exact_matches': ['BoerBurgerBeweging', 'BBB', 'Keijzer']
    },
    'CDA': {
        'multistring_matches': ['Christen-Democratisch Appèl', 'Christen Democraten', 'van Gennip',
                                'Bruins Slot', 'de Jonge', 'van Rij'],
        'substring_matches':  ['Christendemocraten'],
        'exact_matches': ['CDA', 'Bontenbal', 'Vedder', 'Hoekstra', 'Schreinemacher', 'Heijnen']
    },
    'SP': {
        'multistring_matches': ['Socialistische Partij', 'Jimmy Dijk'],
        'substring_matches': [],
        'exact_matches': ['SP', 'Marijnissen']
    },
    'FVD': {
        'multistring_matches': ['Forum voor Democratie', 'van Houwelingen', 'van Meijeren', 'Freek Jansen'],
        'substring_matches': [],
        'exact_matches': ['FVD', 'FvD', 'Baudet', ' Forum']
    },
    'PvdD': {
        'multistring_matches': ['Partij voor de Dieren'],
        'substring_matches': ['PvdD'],
        'exact_matches': ['Ouwehand', 'Kostic']
    },
    'CU': {
        'multistring_matches': ['van Ooijen', 'Carola Schouten'],
        'substring_matches': ['ChristenUnie'],
        'exact_matches': ['CU', 'Bikker', 'Grinwis', 'Staghouwer', 'Adema']
    },
    'SGP': {
        'multistring_matches': ['Staatkundig Gereformeerde Partij', 'van der Staaij'],
        'substring_matches': [],
        'exact_matches': ['SGP', 'Stoffer']
    },
    'DENK': {
        'multistring_matches': ['van Baarle', ' Denk'],
        'substring_matches': [],
        'exact_matches': ['DENK', 'Ergin']
    },
    'Volt': {
        'multistring_matches': [],
        'substring_matches': [],
        'exact_matches': ['Volt', 'Dassen', 'Koekkoek']
    },
    'JA21': {
        'multistring_matches': [],
        'substring_matches': [],
        'exact_matches': ['JA21', 'Eerdmans', 'Nanninga']
    },
    'Bij1': {
        'multistring_matches': [],
        'substring_matches': ['Bij1'],
        'exact_matches': ['Olf', 'Simons']
    },
    'BvNL': {
        'multistring_matches': ['Belang van Nederland', 'van Haga'],
        'substring_matches': [],
        'exact_matches': ['BVNL', 'BvNL']
    },
    'Positions': {
        'multistring_matches': ['minister president', 'de premier', 'voorzitter van de partij', 'voorzitter van de tweede kamer'],
        'substring_matches': ['kamervoorzitter', 'lijsttrekker', 'diplomaat', 'minister', 'staatssecretari',
                              'bewinds', 'politic', 'kamerl', 'fractiel', 'volksvertegenwoord', 'premierskandida', 'formateur'],
        'exact_matches': []
    },
    'Politics': {
        'multistring_matches': ['Tweede Kamer', 'Eerste Kamer', 'Raad van State',
                                'Den Haag', 'politieke partij', 'nationaal belang'],
        'substring_matches': ['staat', 'overheid', 'kabinet', 'demissionair', 'regering', 'parlement', 'democrat', 'politi', 'fractie',
                              'kamervra', ' motie', 'electora', 'kiezer', 'burger', 'wetgev', 'verkies', 'verkiez', 'beleid',
                              'oppositie', 'coalitie', 'grondwet', 'campagne', 'lobby', 'conservati', 'libera', 'binnenhof',
                              'progressie', 'populis', 'nationali', 'formatie', 'samenleving', 'debat', 'parlementair',
                              'zetel'],
        'exact_matches': ['links', 'rechts', 'Kamer']
    },
    'Issues': {
        'multistring_matches': ['beter bestuur'],
        'substring_matches': ['migrat', 'migrant', 'asiel', 'vluchteling', 'bestaanszeker', 'woning', 'vaccin', 'pensioen', 'oorlog', 'wef',
                              'stikstof', 'corona', 'bestuur', 'armoed', 'pensioen', 'belasting', 'klimaat', 'duurzaamheid', 'onderwijs'],
        'exact_matches': ['prijsplafond', 'inflatie', 'energiecris']
    },
	'National': {
		'multistring_matches': ['De Waddeneilanden', 'Noord-Holland', 'Zuid-Holland'],
		'substring_matches': ['Nederland','gemeente', 'Holland', 'Zeeland', 'Brabant', 'Gelderland', 'Utrecht', 'Limburg',
							  'Overijssel', 'Drent', 'Flevoland', 'Friese', 'Friessche', 'Friesland', 'Groning'
                              'provinci', 'stadsbestu', 'Randst'],
		'exact_matches': ['Wadden']
    },
'International': {
		'multistring_matches': ['Europese Unie', 'Verenigde Staten'],
		'substring_matches': ['Europ', 'Amerika', 'Afrika', 'Russisch', 'Aziat', 'Scandinav', 'Oceani'],
		'exact_matches': ['Rusland', 'Azië']
    }
}

nl23_political_words_to_remove = {'denk', 'volt', 'dijk', 'forum'}

nl23_politician_full_names_to_remove = {
    'Geert Wilders', 'Fleur Agema', 'Martin Bosma', 'Dion Graus',
    'Frans Timmermans', 'Esmah Lahlah', 'Jesse Klaver',
    'Dilan Yesilgöz', 'Sophie Hermans', 'Mark Rutte', 'Micky Adriaansens','Mark Harbers', 'Conny Helder', 'Minister van Leeuwen',
    'Geoffrey van Leeuwen', 'Christianne van der Wal',
    'Dennis Wiersma', 'Mariëlle Paul', 'Eric van der Burg',
    'Christophe van der Maat', 'Staatssecretaris de Vries', 'Aukje de Vries',
    'Pieter Omtzigt', 'Nicolien van Vroonhoven',
    'Rob Jetten', 'Jan Paternotte', 'Sigrid Kaag',
    'Caroline van der Plas', 'Mona Keijzer',
    'Steven van Weyenberg', 'Kajsa Ollongren', 'Robbert Dijkgraaf', 'Ernst Kuipers',
    'Pia Dijkstra', 'Franc Weerwind', 'Gunay Uslu', 'Fleur Gräper',
    'Alexandra van Huffelen', 'Hans Vijlbrief',
    'Henri Bontenbal', 'Eline Vedder', 'Wopke Hoekstra',
    'Karien van Gennip', 'Liesje Schreinemacher', 'Hanke Bruins Slot',
    'Hugo de Jonge', 'Marnix van Rij', 'Vivianne Heijnen',
    'Lilian Marijnissen', 'Jimmy Dijk',
    'Thierry Baudet', 'Freek Jansen', 'Pepijn van Houwelingen',
    'Esther Ouwehand', 'Ines Kostic',
    'Mirjam Bikker', 'Pieter Grinwis', 'Carola Schouten',
    'Henk Staghouwer', 'Piet Adema',
    'Maarten van Ooijen', 'Kees van der Staaij', 'Chris Stoffer',
    'Stephan van Baarle', 'Dogukan Ergin',
    'Lauren Dassen', 'Marieke Koekkoek',
    'Joost Eerdmans', 'Annabel Nanninga',
    'Edson Olf', 'Sylvana Simons',
    'Wybren van Haga'
}
nl23_politician_last_names = {
    'Wilders',
    'Timmermans',
    'Yesilgöz',
    'Omtzigt',
    'Jetten',
    'van der Plas',
    'Bontenbal',
    'Marijnissen',
    'Baudet',
    'Ouwehand',
    'Bikker',
    'van der Staaij',
    'van Baarle',
    'Dassen',
    'Eerdmans',
    ' Olf',
    'van Haga'
}
nl23_political_parties = ['PVV', 'GL-PvdA', 'VVD', 'NSC', 'D66', 'BBB', 'CDA', 'SP', 'FVD', 'PvdD', 'CU', 'SGP', 'DENK', 'Volt', 'JA21', 'Bij1', 'BvNL']

nl23_politician_party_OLD = {
    'Geert Wilders': 'PVV',
    'Frans Timmermans': 'GL-PvdA',
    'Dilan Yesilgöz': 'VVD',
    'Pieter Omtzigt': 'NSC',
    'Rob Jetten': 'D66',
    'Caroline van der Plas': 'BBB',
    'Henri Bontebal': 'CDA',
    'Lilian Marijnissen': 'SP',
    'Thierry Baudet': 'FVD',
    'Esther Ouwehand': 'PvdD',
    'Mirjam Bikker': 'CU',
    'Chris Stoffer': 'SGP',
    'Stephan van Baarle': 'DENK',
    'Laurens Dassen': 'Volt',
    'Joost Eerdmans': 'JA21',
    'Edson Olf': 'Bij1',
    'Wybren van Haga': 'BvNL'
}

nl23_politician_party = {
    'Geert_Wilders': 'PVV',
    'Frans_Timmermans': 'GL-PvdA',
    'Dilan_Yesilgoz': 'VVD',
    'Pieter_Omtzigt': 'NSC',
    'Rob_Jetten': 'D66',
    'Caroline_van_der_Plas': 'BBB',
    'Henri_Bontenbal': 'CDA',
    'Lilian_Marijnissen': 'SP',
    'Thierry_Baudet': 'FVD',
    'Esther_Ouwehand': 'PvdD',
    'Mirjam_Bikker': 'CU',
    'Chris_Stoffer': 'SGP',
    'Stephan_van_Baarle': 'DENK',
    'Laurens_Dassen': 'Volt',
    'Joost_Eerdmans': 'JA21',
    'Edson_Olf': 'Bij1',
    'Wybren_van_Haga': 'BvNL'
}


nl23_political_party_colnames = ['PVV_title', 'GL-PvdA_title', 'VVD_title',
       'NSC_title', 'D66_title', 'BBB_title', 'CDA_title', 'SP_title',
       'FVD_title', 'PvdD_title', 'CU_title', 'SGP_title', 'DENK_title',
       'Volt_title', 'JA21_title', 'Bij1_title', 'BvNL_title',
       'PVV_paragraphs', 'GL-PvdA_paragraphs', 'VVD_paragraphs',
       'NSC_paragraphs', 'D66_paragraphs', 'BBB_paragraphs', 'CDA_paragraphs',
       'SP_paragraphs', 'FVD_paragraphs', 'PvdD_paragraphs', 'CU_paragraphs',
       'SGP_paragraphs', 'DENK_paragraphs', 'Volt_paragraphs',
       'JA21_paragraphs', 'Bij1_paragraphs', 'BvNL_paragraphs', 
       'PVV_alt_txt', 'GL-PvdA_alt_txt', 'VVD_alt_txt',
       'NSC_alt_txt', 'D66_alt_txt', 'BBB_alt_txt', 'CDA_alt_txt',
       'SP_alt_txt', 'FVD_alt_txt', 'PvdD_alt_txt', 'CU_alt_txt',
       'SGP_alt_txt', 'DENK_alt_txt', 'Volt_alt_txt', 'JA21_alt_txt',
       'Bij1_alt_txt', 'BvNL_alt_txt']

# Wordlists for GL-PvdA consistency mappings
glpvda_search_list = ['PvdA-GL', 'GL-PvdA', 'GL/PvdA', 'PvdA/GL', 'GroenLinks/PvdA', 'PvdA/GroenLinks', 'GroenLinks-PvdA', 'PvdA-GroenLinks']
glpvda_mapping = 'GL-PvdA'

#######################
######### UK ##########
#######################

uk24_political_words = {
    'Labour_Party': {
        'multistring_matches': [],
        'substring_matches': [],
        'exact_matches': ['LP', 'Labour', 'Starmer', 'Rayner', 'Reeves', 'Streeting', 
                          'Cooper', 'Lammy', 'Ashworth', 'Nandy', 'Thornberry']
    },
    'Conservative_Party': {
        'multistring_matches': ['Conservative Party', 'Rees-Mogg'],
        'substring_matches': [],
        'exact_matches': ['CP', 'Conservatives', 'Tories', 'Tory', 'Sunak', 'Hunt', 'Mordaunt', 
                          'Cleverly', 'Braverman', 'Badenoch', 'Shapps', 'Dowden', 'Truss']
    },
    'Reform_UK': {
        'multistring_matches': ['Reform Party', 'Reform UK'],
        'substring_matches': [],
        'exact_matches': ['RUK', 'Farage', 'Tice', 'Anderson', 'Widdecombe', 'Pochin']
    },
    'Liberal_Democrats': {
        'multistring_matches': ['Lib Dems', 'Liberal Democrats'],
        'substring_matches': ['LibDem'],
        'exact_matches': ['LD', 'Davey', 'Cooper', 'Moran']
    },
    'Green_Party_of_England_and_Wales': {
        'multistring_matches': ['Green Party'],
        'substring_matches': [],
        'exact_matches': ['GP', 'Greens', 'Denyer', 'Ramsay', 'Lucas', 'Berry']
    },
    'Scottish_National_Party': {
        'multistring_matches': ['Scottish National Party'],
        'substring_matches': [],
        'exact_matches': ['SNP', 'Swinney', 'Flynn', 'Yousaf', 'Sturgeon']
    },
    'Sinn_Féin': {
        'multistring_matches': ['Sinn Féin'],
        'substring_matches': [],
        'exact_matches': ['SF', 'McDonald', "O'Neill", 'Adams']
    },
    'Workers_Party_of_Britain': {
        'multistring_matches': ['Workers Party', "Worker's Party"],
        'substring_matches': [],
        'exact_matches': ['WP', 'Galloway']      
    },
    'Plaid_Cymru': {
        'multistring_matches': ['Plaid Cymru'],
        'substring_matches': [],
        'exact_matches': ['PC', 'Iorwerth', 'Price', 'Roberts']
    },
    'Democratic_Unionist_Party': {
        'multistring_matches': ['Democratic Unionist Party', 'Democratic Unionists'],
        'substring_matches': [],
        'exact_matches': ['DUP', 'Robinson', 'Donaldson']
    },
    'Alliance_Party': {
        'multistring_matches': ['Naomi Long'],
        'substring_matches': [],
        'exact_matches': ['Alliance']
    },
    'Ulster_Unionist': {
        'multistring_matches': ['Ulster Unionist', 'Ulster Unionists'],
        'substring_matches': [],
        'exact_matches': ['UU', 'Beattie', 'AP']
    },
    'Scottish_Greens': {
        'multistring_matches': ['Scottish Greens'],
        'substring_matches': [],
        'exact_matches': ['SG', 'Slater', 'Harvie']
    },
    'Social_Democratic_and_Labour': {
        'multistring_matches': ['Social Democratic & Labour', 'Social Democratic and Labour', 'Colum Eastwood'],
        'substring_matches': [],
        'exact_matches': ['SDL']
    },
    'Traditional_Unionist_Voice': {
        'multistring_matches': ['Traditional Unionist Voice'],
        'substring_matches': [],
        'exact_matches': ['TUV', 'Allister']
    },
    'Positions': {
        'multistring_matches': [
            'Prime Minister', 'Chancellor of the Exchequer', 'Leader of the Opposition',
            'Home Secretary', 'Foreign Secretary', 'First Minister', 'Deputy PM', 'Lord Chancellor',
            'junior minister'
        ],
        'substring_matches': [
            'minister', 'secretar', 'candidate', 'leader', 'speaker', 'council', 'chancellor'
        ],
        'exact_matches': ['PM', 'MPs', 'MP', 'MLA']
    },
    'Politics': {
        'multistring_matches': [
            'House of Commons', 'House of Lords', 'Downing Street', 'general election',
            'by-election', 'First Past the Post',  'political party',
            'Shadow Chancellor', 'Shadow Cabinet', 'cabinet reshuffle', 'postal vote', 'exit poll',
            'swing seat', 'election day', 'poll of polls'
        ],
        'substring_matches': [
            'parliament', 'government', 'opposition', 'cabinet', 'democra', 'elect', 'voting', 'vote', 'campaign',
            'polic', 'coalition', 'referendum', 'lobby', 'debat', 'bill', 'law', 'Brexit',
            'ballot', 'turnout', 'poll', 'constitution', 'constituency', 'manifesto'
        ],
        'exact_matches': ['Whitehall', 'PMQs', 'Westminster']
    },
    'Issues': {
        'multistring_matches': [
            'NHS crisis', 'cost of living', 'climate change', 'housing crisis', 'small boats',
            'Rwanda plan', 'tax cuts', 'public services', 'net zero', 'defence spending',
            'New Deal for Working People', 'Green New Deal', 'Independence referendum', 'Welsh independence',
            'Scottish independence', 'energy bills', 'childcare costs', 'public sector pay', 'rail strikes',
            'social care crisis', 'border control', 'tech regulation', 'AI Act', 'European Union'
        ],
        'substring_matches': [
            'econom', 'immigra', 'health', 'energy', 'education', 'tax', 'polic', 'welfare', 'pension',
            'inflation', 'strike', 'union', 'debt', 'austerity', 'privatisation', 'crisis', 'crises'
        ],
        'exact_matches': ['NHS', 'GDP', 'AI']
    },
    'National': {
        'multistring_matches': ['United Kingdom', 'Great Britain', 'Northern Ireland', 'Republic of Ireland'],
        'substring_matches': ['Engl', 'provinc', 'district', 'state', 'Britain', 'Scottish', 'Welsh'],
        'exact_matches': ['Wales', 'Ireland', 'Scotland', 'UK', 'GB', 'NI']
    },
	'International': {
        'multistring_matches': ['European Union', 'United States'],
        'substring_matches': ['Engl', 'America', 'Africa', 'Scandinavia', 'Asia', 'Latin', 'Oceani'],
        'exact_matches': ['EU', 'USA']
	}
}


uk24_politician_party_OLD = {
    'Keir Starmer': 'Labour_Party',                     # Labour / 411 
    'Rishi Sunak': 'Conservative_Party',                # Conservative / 121
    'Nigel Farage': 'Reform_UK',                        # Reform UK / 5
    'Ed Davey': 'Liberal_Democrats',                    # Liberal Democrats / 72
    'Carla Denyer': 'Green_Party_of_England_and_Wales', # Green Party of England and Wales / 4
    'Adrian Ramsay': 'Green_Party_of_England_and_Wales',# Green Party of England and Wales / 4
    'John Swinney': 'Scottish_National_Party',          # Scottish National Party / 9
    'Mary Lou McDonald': 'Sinn_Fein',                   # Sinn Féin / 7
    'George Galloway': 'Workers_Party_of_Britain',      # Workers Party / 0 (new)
    'Rhun ap Iorwerth': 'Plaid_Cymru',                  # Plaid Cymru / 4
    'Gavin Robinson': 'Democratic_Unionist_Party',      # Democratic Unionist / 5
    'Naomi Long': 'Alliance_Party',                     # Alliance Party / 1
    'Doug Beattie': 'Ulster_Unionist_Party',            # Ulster Unionist Party / 1
    'Patrick Harvie': 'Scottish_Greens',                # Scottish Greens / 0
    'Lorna Slater': 'Scottish_Greens',                  # Scottish Greens / 0
    'Colum Eastwood': 'Social_Democratic_and_Labour',   # Social Democratic & Labour / 2
    'Jim Allister': 'Traditional_Unionist_Voice'        # Traditional Unionist Vote / 1
}


uk24_politician_party = {
    'Keir_Starmer': 'Labour_Party',                     # Labour / 411 
    'Rishi_Sunak': 'Conservative_Party',                # Conservative / 121
    'Ed_Davey': 'Liberal_Democrats',                    # Liberal Democrats / 72
    'John_Swinney': 'Scottish_National_Party',          # Scottish National Party / 9
    'Mary_Lou_McDonald': 'Sinn_Féin',                   # Sinn Féin / 7
    'Nigel_Farage': 'Reform_UK',                        # Reform UK / 5
    'Gavin_Robinson': 'Democratic_Unionist_Party',      # Democratic Unionist / 5
    'Carla_Denyer': 'Green_Party_of_England_and_Wales', # Green Party of England and Wales / 4
    'Adrian_Ramsay': 'Green_Party_of_England_and_Wales',# Green Party of England and Wales / 4
    'Rhun_ap_Iorwerth': 'Plaid_Cymru',                  # Plaid Cymru / 4
    'Colum_Eastwood': 'Social_Democratic_and_Labour',   # Social Democratic & Labour / 2
    'Jim_Allister': 'Traditional_Unionist_Voice',        # Traditional Unionist Vote / 1
    'Naomi_Long': 'Alliance_Party',                     # Alliance Party / 1
    'Doug_Beattie': 'Ulster_Unionist',                  # Ulster Unionist Party / 1
    'George_Galloway': 'Workers_Party_of_Britain',      # Workers Party / 0 (new)
    'Patrick_Harvie': 'Scottish_Greens',                # Scottish Greens / 0
    'Lorna_Slater': 'Scottish_Greens'                  # Scottish Greens / 0
}

uk24_politician_last_names = {
    'Starmer',
    'Sunak',
    'Farage',
    'Davey',
    'Denyer',
    'Ramsay',
    'Swinney',
    'McDonald',
    'Galloway',
    'Iorwerth',
    'Robinson',
    'Long',
    'Beattie',
    'Harvie',
    'Slater',
    'Eastwood',
    'Allister'
}
uk24_parties = [
    'Labour_Party', 
    'Conservative_Party', 
    'Reform_UK', 
    'Liberal_Democrats', 
    'Green_Party_of_England_and_Wales',
    'Scottish_National_Party',
    'Sinn_Fein',
    'Workers_Party_of_Britain',
    'Plaid_Cymru',
    'Democratic_Unionist_Party',
    'Alliance_Party',
    'Ulster_Unionist_Party',
    'Scottish_Greens',
    'Social_Democratic_and_Labour',
    'Traditional_Unionist_Voice'
]